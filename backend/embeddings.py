
from typing import Optional, Tuple
import numpy as np
from PIL import Image
from pathlib import Path
from .config import EMBEDDING_BACKEND
from .utils import safe_open_image

# Use TensorFlow for embeddings, fallback to HOG features
def _tf_encoder(img: Image.Image) -> Optional[np.ndarray]:
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import mobilenet_v2, resnet50
        from tensorflow.keras import Model

        # cache model on function attribute
        if not hasattr(_tf_encoder, "_cache"):
            base = mobilenet_v2.MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
            # Alternative lightweight choices: EfficientNetB0, NASNetMobile
            size = (224, 224)
            setattr(_tf_encoder, "_cache", (base, size))
        else:
            base, size = getattr(_tf_encoder, "_cache")

        # preprocess
        arr = np.array(img.convert("RGB").resize(size), dtype=np.float32)
        x = mobilenet_v2.preprocess_input(arr)
        x = np.expand_dims(x, axis=0)
        feat = base.predict(x, verbose=0).reshape(-1)
        feat = feat / (np.linalg.norm(feat) + 1e-9)
        return feat.astype(np.float32)
    except ImportError:
        return None
    except Exception:
        return None

def _hog_encoder(img: Image.Image) -> np.ndarray:
    # HOG + color hist fallback
    import numpy as np
    from skimage.color import rgb2gray
    from skimage.transform import resize
    from skimage.feature import hog

    arr = np.array(img)
    gray = rgb2gray(arr)
    gray = resize(gray, (256, 256), anti_aliasing=True)
    hog_vec = hog(gray, pixels_per_cell=(16,16), cells_per_block=(2,2), orientations=9, block_norm="L2-Hys", feature_vector=True)
    # simple 8-bin per channel histogram
    hist_r, _ = np.histogram(arr[...,0], bins=8, range=(0,255), density=True)
    hist_g, _ = np.histogram(arr[...,1], bins=8, range=(0,255), density=True)
    hist_b, _ = np.histogram(arr[...,2], bins=8, range=(0,255), density=True)
    feat = np.concatenate([hog_vec, hist_r, hist_g, hist_b]).astype(np.float32)
    feat = feat / (np.linalg.norm(feat) + 1e-9)
    return feat

def compute_embedding(path: Path) -> np.ndarray:
    img = safe_open_image(path).convert("RGB")
    enc = None
    if EMBEDDING_BACKEND in ("auto", "tf", "tensorflow"):
        enc = _tf_encoder(img)
    if enc is None:
        enc = _hog_encoder(img)
    return enc
