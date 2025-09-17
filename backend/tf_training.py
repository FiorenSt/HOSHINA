from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Thread, Lock
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .db import engine, DatasetItem, Prediction, ClassDef
from .utils import safe_open_image
from .config import STORE_DIR
from sqlmodel import select, Session


@dataclass
class TrainingStatus:
    running: bool = False
    stage: str = "idle"  # idle | preparing | training | predicting | done | error | cancelled
    epoch: int = 0
    total_epochs: int = 0
    steps_per_epoch: int = 0
    total_steps: int = 0
    loss: float = 0.0
    acc: float = 0.0
    val_loss: Optional[float] = None
    val_acc: Optional[float] = None
    message: str = ""
    model_name: Optional[str] = None
    params: Dict[str, str] = field(default_factory=dict)
    device: Optional[str] = None
    gpu: Optional[str] = None
    history: Dict[str, List[float]] = field(default_factory=lambda: {"epoch": [], "loss": [], "acc": [], "val_loss": [], "val_acc": []})


_status = TrainingStatus()
_status_lock = Lock()
_worker: Optional[Thread] = None
_cancel_flag = False


def _set_status(**kwargs):
    global _status
    with _status_lock:
        for k, v in kwargs.items():
            setattr(_status, k, v)


def get_status() -> TrainingStatus:
    with _status_lock:
        # return a shallow copy to avoid mutation
        return TrainingStatus(**_status.__dict__)


def cancel():
    global _cancel_flag
    _cancel_flag = True


def available_models() -> List[Dict[str, str]]:
    return [
        {"id": "mobilenet_v2", "name": "MobileNetV2 (fast, small, 224px)", "input": "224x224"},
        {"id": "efficientnet_b0", "name": "EfficientNetB0 (balanced, 224px)", "input": "224x224"},
        {"id": "resnet50", "name": "ResNet50 (heavier, 224px)", "input": "224x224"},
    ]


def _build_model(model_id: str, num_classes: int, loss: str = "sparse_categorical_crossentropy"):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    # Fixed spatial size for all backbones in this app
    input_size = (224, 224)
    # Bind a 3-channel input tensor to avoid accidental 1-channel graphs
    inputs = layers.Input(shape=(input_size[0], input_size[1], 3))

    if model_id == "custom":
        # Build from a saved custom graph spec
        model, preprocess, input_size = _build_model_from_saved_graph(num_classes)
        return model, preprocess, input_size
    elif model_id == "mobilenet_v2":
        from tensorflow.keras.applications import mobilenet_v2 as app
        preprocess = app.preprocess_input
        base = app.MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_tensor=inputs)
    elif model_id == "efficientnet_b0":
        from tensorflow.keras.applications import efficientnet as app
        preprocess = app.preprocess_input
        base = app.EfficientNetB0(weights="imagenet", include_top=False, pooling="avg", input_tensor=inputs)
    elif model_id == "resnet50":
        from tensorflow.keras.applications import resnet50 as app
        preprocess = app.preprocess_input
        base = app.ResNet50(weights="imagenet", include_top=False, pooling="avg", input_tensor=inputs)
    else:
        # default
        from tensorflow.keras.applications import mobilenet_v2 as app
        preprocess = app.preprocess_input
        base = app.MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_tensor=inputs)

    # quick transfer: freeze backbone by default
    base.trainable = False
    x = base.output
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model, preprocess, input_size


# ===== Custom graph-based model builder =====
def _custom_graph_path() -> Path:
    from .config import STORE_DIR
    p = STORE_DIR / "model" / "tf"
    p.mkdir(parents=True, exist_ok=True)
    return p / "custom_model.json"


def get_saved_custom_graph() -> Optional[dict]:
    import json
    fp = _custom_graph_path()
    if not fp.exists():
        return None
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_custom_graph(graph: dict) -> None:
    import json
    fp = _custom_graph_path()
    fp.write_text(json.dumps(graph, indent=2), encoding="utf-8")


def _build_model_from_saved_graph(num_classes: int):
    import tensorflow as tf
    from tensorflow.keras.applications import mobilenet_v2 as app
    graph = get_saved_custom_graph()
    if not graph:
        raise RuntimeError("No custom model graph saved. Open Model Builder to create one.")
    model = build_model_from_graph(graph, num_classes=num_classes)
    # Use ImageNet-style preprocessing by default
    preprocess = app.preprocess_input
    input_size = (224, 224)
    return model, preprocess, input_size


def _ensure_dag(nodes: List[dict], edges: List[dict]) -> None:
    # Simple DAG validation: detect cycles via DFS
    from collections import defaultdict
    adj = defaultdict(list)
    ids = {n["id"] for n in nodes}
    for e in edges:
        u = e.get("from"); v = e.get("to")
        if u not in ids or v not in ids:
            raise ValueError("Edge references unknown node id")
        adj[u].append(v)
    visited = {}
    def dfs(u):
        state = visited.get(u, 0)
        if state == 1:
            raise ValueError("Cycle detected in model graph")
        if state == 2:
            return
        visited[u] = 1
        for v in adj[u]:
            dfs(v)
        visited[u] = 2
    for n in ids:
        if visited.get(n, 0) == 0:
            dfs(n)


def build_model_from_graph(graph_spec: dict, num_classes: int, loss: str = "sparse_categorical_crossentropy"):
    """Build a tf.keras Model from a simple graph specification.

    graph_spec: {
      nodes: [{ id: str, type: str, params?: dict }],
      edges: [{ from: str, to: str }],
      input_shape?: [h, w, c]
    }

    Supported node types: Input, Conv2D, SeparableConv2D, MaxPool2D, GlobalAveragePooling2D,
    BatchNormalization, Activation, Dropout, Flatten, Dense, Add, Concat.
    The final classification layer (Dense softmax) is appended automatically.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    nodes = graph_spec.get("nodes") or []
    edges = graph_spec.get("edges") or []
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("Invalid graph_spec: nodes/edges must be lists")
    # Validate DAG
    _ensure_dag(nodes, edges)

    # Build lookup tables
    node_by_id = {str(n.get("id")): n for n in nodes}
    inputs_to = {nid: [] for nid in node_by_id.keys()}
    outputs_of = {nid: [] for nid in node_by_id.keys()}
    for e in edges:
        u = str(e.get("from")); v = str(e.get("to"))
        if u in node_by_id and v in node_by_id:
            inputs_to[v].append(u)
            outputs_of[u].append(v)

    # Determine input shape
    ish = graph_spec.get("input_shape")
    if (not ish) or len(ish) != 3:
        ish = [224, 224, 3]

    # Create Keras tensors per node
    tensors: Dict[str, tf.Tensor] = {}
    input_node_id = None
    for nid, n in node_by_id.items():
        if (n.get("type") or "").lower() == "input":
            input_node_id = nid
            break
    if input_node_id is None:
        # Implicit input node
        input_node_id = "input"
        node_by_id[input_node_id] = {"id": input_node_id, "type": "Input", "params": {}}
        inputs_to[input_node_id] = inputs_to.get(input_node_id, [])
        outputs_of[input_node_id] = outputs_of.get(input_node_id, [])

    keras_input = layers.Input(shape=tuple(ish))
    tensors[input_node_id] = keras_input

    # Topological order: simple Kahn based on inputs_to
    indeg = {nid: len(inputs_to.get(nid, [])) for nid in node_by_id.keys()}
    from collections import deque
    q = deque([nid for nid, d in indeg.items() if d == 0])
    order = []
    while q:
        u = q.popleft(); order.append(u)
        for v in outputs_of.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    # Build layers following the order
    for nid in order:
        if nid == input_node_id:
            continue
        node = node_by_id[nid]
        ntype = (node.get("type") or "").strip()
        params = node.get("params") or {}
        inbound = inputs_to.get(nid, [])
        inbound_tensors = [tensors[i] for i in inbound if i in tensors]
        x = inbound_tensors[0] if inbound_tensors else list(tensors.values())[0]

        if ntype == "Conv2D":
            x = layers.Conv2D(filters=int(params.get("filters", 32)), kernel_size=int(params.get("kernel_size", 3)), strides=int(params.get("strides", 1)), padding=params.get("padding", "same"), activation=params.get("activation", None))(x)
        elif ntype == "SeparableConv2D":
            x = layers.SeparableConv2D(filters=int(params.get("filters", 32)), kernel_size=int(params.get("kernel_size", 3)), strides=int(params.get("strides", 1)), padding=params.get("padding", "same"), activation=params.get("activation", None))(x)
        elif ntype in {"MaxPool", "MaxPool2D", "MaxPooling2D"}:
            x = layers.MaxPool2D(pool_size=int(params.get("pool_size", 2)), strides=int(params.get("strides", 2)), padding=params.get("padding", "same"))(x)
        elif ntype in {"GAP", "GlobalAveragePooling2D"}:
            x = layers.GlobalAveragePooling2D()(x)
        elif ntype in {"BN", "BatchNormalization"}:
            x = layers.BatchNormalization()(x)
        elif ntype in {"Act", "Activation"}:
            x = layers.Activation(params.get("activation", "relu"))(x)
        elif ntype == "Dropout":
            x = layers.Dropout(float(params.get("rate", 0.2)))(x)
        elif ntype == "Flatten":
            x = layers.Flatten()(x)
        elif ntype == "Dense":
            x = layers.Dense(int(params.get("units", 128)), activation=params.get("activation", "relu"))(x)
        elif ntype in {"Add", "add"}:
            # If multiple inbound tensors exist, add them; otherwise passthrough
            if len(inbound_tensors) >= 2:
                x = layers.Add()(inbound_tensors)
            else:
                x = inbound_tensors[0] if inbound_tensors else x
        elif ntype in {"Concat", "Concatenate"}:
            if len(inbound_tensors) >= 2:
                x = layers.Concatenate(axis=params.get("axis", -1))(inbound_tensors)
            else:
                x = inbound_tensors[0] if inbound_tensors else x
        elif ntype in {"Output", "ClassifierHead"}:
            # handled after graph construction; keep passthrough
            x = x
        else:
            raise ValueError(f"Unsupported node type: {ntype}")

        tensors[nid] = x

    # Determine terminal tensor
    terminals = [nid for nid in node_by_id.keys() if len(outputs_of.get(nid, [])) == 0]
    last_id = terminals[0] if terminals else order[-1]
    last_tensor = tensors[last_id]

    # If last tensor is 4D, pool to vector
    if len(last_tensor.shape) == 4:
        last_tensor = layers.GlobalAveragePooling2D()(last_tensor)
    # Append classifier head
    outputs = layers.Dense(num_classes, activation="softmax")(last_tensor)
    model = models.Model(inputs=keras_input, outputs=outputs)
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


def _compose_triplet_array(p: Path, input_size: Tuple[int, int]) -> np.ndarray:
    """Compose a 3-channel array from target/ref/diff triplet.

    - Channel R: target
    - Channel G: ref (or zeros)
    - Channel B: diff (or zeros)
    """
    # Derive base
    name_lower = p.name.lower()
    base = None
    for suf in ["_target.fits", "_ref.fits", "_diff.fits"]:
        if name_lower.endswith(suf):
            base = p.name[: -len(suf)]
            break
    if base is None:
        base = p.stem

    expected = {
        "target": p.with_name(f"{base}_target.fits"),
        "ref": p.with_name(f"{base}_ref.fits"),
        "diff": p.with_name(f"{base}_diff.fits"),
    }

    def load_gray(path: Path) -> np.ndarray:
        try:
            img = safe_open_image(path).convert("L").resize(input_size)
            arr = np.asarray(img, dtype=np.float32)
            return arr
        except Exception:
            return np.zeros(input_size, dtype=np.float32)

    r = load_gray(expected["target"]) if expected["target"].exists() else np.zeros(input_size, dtype=np.float32)
    g = load_gray(expected["ref"]) if expected["ref"].exists() else np.zeros(input_size, dtype=np.float32)
    b = load_gray(expected["diff"]) if expected["diff"].exists() else np.zeros(input_size, dtype=np.float32)
    rgb = np.stack([r, g, b], axis=-1)
    return rgb


def _make_dataset(
    paths: List[Path],
    labels: List[int],
    input_size: Tuple[int, int],
    preprocess: Callable,
    augment: bool,
    batch_size: int,
    input_mode: str,
    single_role: Optional[str] = None,
    one_hot: bool = False,
    num_classes: Optional[int] = None,
    sample_weight_map: Optional[Dict[int, float]] = None,
):
    import tensorflow as tf

    # Convert to tensors once for efficient pipeline construction
    path_strs = [str(p) for p in paths]
    ds = tf.data.Dataset.from_tensor_slices((path_strs, np.array(labels, dtype=np.int64)))

    # Options: allow non-deterministic for throughput and enable optimizations
    opts = tf.data.Options()
    try:
        opts.experimental_deterministic = False  # best-effort ordering
        opts.experimental_optimization.apply_default_optimizations = True
        opts.experimental_optimization.autotune_buffers = True
        opts.experimental_optimization.autotune_map_parallelism = True
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.parallel_batch = True
        opts.experimental_optimization.shutdown_quietly = True
    except Exception:
        pass
    ds = ds.with_options(opts)

    target_h, target_w = int(input_size[0]), int(input_size[1])

    def _load_array_py(p_bytes: bytes) -> np.ndarray:
        # Python side loader used via tf.py_function
        try:
            p = Path(p_bytes.decode("utf-8"))
        except Exception:
            return np.zeros((target_h, target_w, 3), dtype=np.float32)

        try:
            if input_mode == "triplet":
                arr = _compose_triplet_array(p, (target_h, target_w)).astype(np.float32)
            else:
                role = (single_role or "").lower()
                name_lower = p.name.lower()
                base = p.stem
                for suf in ["_target.fits", "_ref.fits", "_diff.fits"]:
                    if name_lower.endswith(suf):
                        base = p.name[: -len(suf)]
                        break
                if role in {"target", "ref", "diff"}:
                    rp = p.with_name(f"{base}_{role}.fits")
                    try:
                        img = safe_open_image(rp if rp.exists() else p).convert("L").resize((target_h, target_w))
                        g = np.asarray(img, dtype=np.float32)
                    except Exception:
                        g = np.zeros((target_h, target_w), dtype=np.float32)
                    arr = np.stack([g, g, g], axis=-1)
                else:
                    img = safe_open_image(p).convert("RGB").resize((target_h, target_w))
                    arr = np.asarray(img, dtype=np.float32)
        except Exception:
            arr = np.zeros((target_h, target_w, 3), dtype=np.float32)
        # Ensure correct shape/dtype
        if arr.ndim != 3 or arr.shape[-1] != 3:
            arr = np.zeros((target_h, target_w, 3), dtype=np.float32)
        return arr

    def _map_fn(p_str, y):
        x = tf.py_function(_load_array_py, [p_str], Tout=tf.float32)
        x.set_shape((target_h, target_w, 3))
        # Apply model-specific preprocessing as a TensorFlow op
        x = preprocess(x)
        if one_hot:
            depth = int(num_classes or 0)
            y_oh = tf.one_hot(tf.cast(y, tf.int32), depth=depth, dtype=tf.float32)
            if sample_weight_map is not None and depth > 0:
                weights = tf.constant([sample_weight_map.get(i, 1.0) for i in range(depth)], dtype=tf.float32)
                sw = tf.gather(weights, tf.cast(y, tf.int32))
                return x, y_oh, sw
            return x, y_oh
        return x, tf.cast(y, tf.int64)

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        # Only allow rotations (0/90/180/270) and mirroring (horizontal/vertical).
        # Since triplets are composed into a single 3-channel tensor, these ops
        # apply the exact same transform to all images in the triplet.
        def _rot_mirror(x, y, *rest):
            k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
            x = tf.image.rot90(x, k)
            do_h = tf.random.uniform([]) < 0.5
            x = tf.cond(do_h, lambda: tf.image.flip_left_right(x), lambda: x)
            do_v = tf.random.uniform([]) < 0.5
            x = tf.cond(do_v, lambda: tf.image.flip_up_down(x), lambda: x)
            if len(rest) == 1:
                return x, y, rest[0]
            return x, y
        ds = ds.map(_rot_mirror, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.shuffle(buffer_size=min(2048, max(32, len(paths))))
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def start_training(
    model_id: str = "mobilenet_v2",
    epochs: int = 3,
    batch_size: int = 32,
    augment: bool = False,
    input_mode: str = "single",
    class_map: Optional[Dict[str, int]] = None,
    split_pct: int = 85,
    split_strategy: str = "natural",
    single_role: str = "target",
    loss: str = "sparse_categorical_crossentropy",
    class_weight: Optional[Dict[str, float]] = None,
):
    global _worker, _cancel_flag

    if _worker and _worker.is_alive():
        raise RuntimeError("Training already running")

    _cancel_flag = False

    def work():
        import tensorflow as tf

        # Device / GPU info
        try:
            gpus = tf.config.list_physical_devices('GPU')
            gpu_name = gpus[0].name if gpus else None
        except Exception:
            gpu_name = None

        params = {"batch_size": str(batch_size), "augment": str(augment), "input_mode": input_mode, "split_pct": str(split_pct), "split_strategy": split_strategy, "loss": loss}
        if input_mode == "single":
            params["single_role"] = single_role
        if class_map:
            params["class_map"] = str(class_map)
        if class_weight:
            params["class_weight"] = str(class_weight)
        _set_status(running=True, stage="preparing", epoch=0, total_epochs=epochs, model_name=model_id, params=params, device=tf.test.gpu_device_name() or "CPU", gpu=gpu_name)
        try:
            # Fetch classes and labeled items
            with Session(engine) as session:
                cls = session.exec(select(ClassDef).order_by(ClassDef.order)).all()
                # If class_map is provided: filter dataset to those labels and use provided numeric mapping
                if class_map:
                    # Keep only classes that appear in class_map
                    class_names = [c.name for c in cls if c.name in class_map]
                    class_to_idx = {name: int(class_map[name]) for name in class_names}
                    # Validate unique integer targets
                    inv = {}
                    for k, v in class_to_idx.items():
                        if v in inv:
                            raise ValueError(f"Duplicate target value {v} for labels '{inv[v]}' and '{k}'")
                        inv[v] = k
                    # Sort by numeric target for deterministic output width
                    ordered = sorted(class_to_idx.items(), key=lambda kv: kv[1])
                    class_to_idx = {k: v for k, v in ordered}
                else:
                    class_names = [c.name for c in cls]
                    class_to_idx = {c: i for i, c in enumerate(class_names)}

                rows = session.exec(select(DatasetItem).where(DatasetItem.label.is_not(None))).all()
                if len(rows) < 2:
                    _set_status(stage="error", running=False, message="Need at least 2 labeled items")
                    return
                # Build lists
                paths: List[Path] = []
                labels: List[int] = []
                if input_mode == "triplet":
                    # Use labeled targets as anchors
                    for r in rows:
                        p = Path(r.path)
                        nm = p.name.lower()
                        if nm.endswith("_target.fits") and r.label in class_to_idx:
                            paths.append(p)
                            labels.append(class_to_idx[r.label])
                else:
                    role = (single_role or "").lower()
                    if role in {"target", "ref", "diff"}:
                        # Anchor on labeled targets, but load the chosen role path per base
                        for r in rows:
                            p = Path(r.path)
                            nm = p.name.lower()
                            if nm.endswith("_target.fits") and r.label in class_to_idx:
                                base = p.name[: -len("_target.fits")]
                                rp = p.with_name(f"{base}_{role}.fits")
                                # Only include if the file exists (fallback to target for role=target)
                                if rp.exists() or role == "target":
                                    paths.append(rp if rp.exists() else p)
                                    labels.append(class_to_idx[r.label])
                    else:
                        # Generic single-image case: include all labeled items as-is
                        for r in rows:
                            if r.label in class_to_idx:
                                paths.append(Path(r.path))
                                labels.append(class_to_idx[r.label])
                if len(paths) < 2 or len(set(labels)) < 2:
                    _set_status(stage="error", running=False, message="Not enough data after applying label mapping; need at least 2 classes")
                    return

            # Split train/val according to strategy
            def stratified_indices(labels_list, train_ratio):
                labels_arr = np.array(labels_list)
                tr, va = [], []
                rng = np.random.default_rng(42)
                for c in np.unique(labels_arr):
                    idxs = np.where(labels_arr == c)[0]
                    rng.shuffle(idxs)
                    k = int(len(idxs) * train_ratio)
                    tr.extend(idxs[:k].tolist())
                    va.extend(idxs[k:].tolist())
                return np.array(tr), np.array(va)

            train_ratio = max(0.5, min(0.95, float(split_pct) / 100.0))
            if split_strategy == "balanced":
                # downsample each class to the size of the smallest class, then split equally
                labels_arr = np.array(labels)
                classes = np.unique(labels_arr)
                rng = np.random.default_rng(42)
                per_class = min((labels_arr == c).sum() for c in classes)
                sel_idxs = []
                for c in classes:
                    idxs = np.where(labels_arr == c)[0]
                    rng.shuffle(idxs)
                    sel_idxs.extend(idxs[:per_class].tolist())
                sel_idxs = np.array(sel_idxs)
                rng.shuffle(sel_idxs)
                k = int(len(sel_idxs) * train_ratio)
                tr_idx, va_idx = sel_idxs[:k], sel_idxs[k:]
            elif split_strategy == "stratified":
                tr_idx, va_idx = stratified_indices(labels, train_ratio)
            else:
                idx = np.arange(len(paths))
                rng = np.random.default_rng(42)
                rng.shuffle(idx)
                k = int(train_ratio * len(idx))
                tr_idx, va_idx = idx[:k], idx[k:]

            # Keras Dense output requires consecutive class indices starting at 0; remap if custom targets are sparse
            unique_targets = sorted(set(labels))
            remap = {t: i for i, t in enumerate(unique_targets)}
            inv_remap = {i: t for t, i in remap.items()}
            labels = [remap[t] for t in labels]
            # Build model with chosen loss
            model, preprocess, input_size = _build_model(model_id, num_classes=len(unique_targets), loss=loss)

            # Prepare class weighting map keyed by contiguous indices if provided
            cw_map_idx: Optional[Dict[int, float]] = None
            if class_weight:
                try:
                    # Convert provided map (target index as string) to contiguous index
                    cw_map_idx = {}
                    for k, v in class_weight.items():
                        original_idx = int(k)
                        if original_idx in remap:
                            cw_map_idx[remap[original_idx]] = float(v)
                except Exception:
                    cw_map_idx = None

            # If categorical loss is selected, emit one-hot and optionally sample weights
            use_one_hot = loss in {"categorical_crossentropy", "categorical_focal_crossentropy"}
            ds_tr = _make_dataset(
                [paths[i] for i in tr_idx],
                [labels[i] for i in tr_idx],
                input_size,
                preprocess,
                augment,
                batch_size,
                input_mode,
                single_role=single_role,
                one_hot=use_one_hot,
                num_classes=len(unique_targets),
                sample_weight_map=cw_map_idx if use_one_hot else None,
            )
            ds_va = _make_dataset(
                [paths[i] for i in va_idx],
                [labels[i] for i in va_idx],
                input_size,
                preprocess,
                False,
                batch_size,
                input_mode,
                single_role=single_role,
                one_hot=use_one_hot,
                num_classes=len(unique_targets),
                sample_weight_map=None,
            )

            class EpochProgress(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    if logs is None:
                        logs = {}
                    # append history
                    with _status_lock:
                        _status.history["epoch"].append(epoch + 1)
                        _status.history["loss"].append(float(logs.get("loss", 0.0)))
                        _status.history["acc"].append(float(logs.get("accuracy", 0.0)))
                        _status.history["val_loss"].append(float(logs.get("val_loss", 0.0)) if "val_loss" in logs else None)
                        _status.history["val_acc"].append(float(logs.get("val_accuracy", 0.0)) if "val_accuracy" in logs else None)
                    _set_status(stage="training", epoch=epoch + 1, loss=float(logs.get("loss", 0.0)), acc=float(logs.get("accuracy", 0.0)), val_loss=float(logs.get("val_loss", 0.0)) if "val_loss" in logs else None, val_acc=float(logs.get("val_accuracy", 0.0)) if "val_accuracy" in logs else None)
                    if _cancel_flag:
                        self.model.stop_training = True

                def on_train_batch_end(self, batch, logs=None):
                    if _cancel_flag:
                        self.model.stop_training = True

            _set_status(stage="training", epoch=0)
            fit_kwargs = {}
            # For sparse losses, we can pass class_weight directly
            if not use_one_hot and class_weight:
                # Remap class_weight keys to contiguous indices
                try:
                    cw_fit = {int(remap[int(k)]): float(v) for k, v in class_weight.items() if int(k) in remap}
                    if cw_fit:
                        fit_kwargs["class_weight"] = cw_fit
                except Exception:
                    pass
            model.fit(ds_tr, validation_data=ds_va, epochs=epochs, callbacks=[EpochProgress()], verbose=0, **fit_kwargs)

            if _cancel_flag:
                _set_status(stage="cancelled", running=False, message="Training cancelled")
                return

            # Save model
            model_dir = STORE_DIR / "model" / "tf"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "model.keras"
            model.save(model_path)
            # Stop here: do not auto-run predictions after training. Leave model ready for prediction.
            _set_status(stage="done", running=False, message="Model trained. Ready for prediction")
        except Exception as e:
            _set_status(stage="error", running=False, message=str(e))

    _worker = Thread(target=work, daemon=True)
    _worker.start()


def list_options() -> Dict[str, object]:
    # Include custom model option if a graph exists (or always include with a note)
    models = available_models()
    custom_present = get_saved_custom_graph() is not None
    models = models + ([{"id": "custom", "name": "Custom (Model Builder)", "input": "224x224"}])
    return {
        "models": models,
        "input_modes": ["single", "triplet"],
        "losses": [
            {"id": "sparse_categorical_crossentropy", "name": "Sparse Categorical Crossentropy"},
            {"id": "categorical_crossentropy", "name": "Categorical Crossentropy"}
        ],
        "defaults": {"model": "mobilenet_v2", "epochs": 3, "batch_size": 32, "augment": False, "input_mode": "single", "single_role": "target", "loss": "sparse_categorical_crossentropy"},
        "custom_available": bool(custom_present),
        "notes": "Images from FITS are ZScaled already; standard ImageNet preprocessing is applied per model. Triplet mode maps target/ref/diff to R/G/B channels.",
    }


