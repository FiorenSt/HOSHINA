#!/usr/bin/env python3
import argparse
import statistics
import time
from pathlib import Path
from typing import List, Tuple
import concurrent.futures as futures

import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from PIL import Image

from backend.db import Session, engine, DatasetItem
from backend.utils import safe_open_image


def time_fn(fn, repeats: int = 1) -> List[float]:
    durations = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        durations.append((time.perf_counter() - t0) * 1000.0)
    return durations


def compose_triplet(target: Path, ref: Path, diff: Path, size: int = 256) -> Image.Image:
    panels: List[Image.Image] = []
    for p in [target, ref, diff]:
        if p and p.exists():
            try:
                img = safe_open_image(p).convert("RGB")
            except Exception:
                img = Image.new("RGB", (size, size), color=(0, 0, 0))
        else:
            img = Image.new("RGB", (size, size), color=(0, 0, 0))
        img.thumbnail((size, size))
        canvas = Image.new("RGB", (size, size), color=(0, 0, 0))
        x = (size - img.width) // 2
        y = (size - img.height) // 2
        canvas.paste(img, (x, y))
        panels.append(canvas)

    out = Image.new("RGB", (size * 3, size), color=(0, 0, 0))
    for i, panel in enumerate(panels):
        out.paste(panel, (i * size, 0))
    return out


def summarize(name: str, timings_ms: List[float]) -> None:
    if not timings_ms:
        print(f"{name}: no samples")
        return
    p95 = np.percentile(timings_ms, 95)
    print(
        f"{name}: n={len(timings_ms)} | avg={statistics.mean(timings_ms):.1f} ms | "
        f"median={statistics.median(timings_ms):.1f} ms | p95={p95:.1f} ms"
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark on-the-fly FITS rendering")
    parser.add_argument("--limit", type=int, default=50, help="Number of base triplets to sample")
    parser.add_argument("--size", type=int, default=256, help="Thumbnail size")
    parser.add_argument("--warmups", type=int, default=1, help="Warmup repeats per item")
    parser.add_argument("--repeats", type=int, default=1, help="Measured repeats per item")
    parser.add_argument("--grid", type=int, default=0, help="Simulate composing this many triplets concurrently (0 to skip)")
    parser.add_argument("--concurrency", type=int, default=8, help="Number of worker threads for grid simulation")
    args = parser.parse_args()

    with Session(engine) as session:
        items = session.query(DatasetItem).all()

    from collections import defaultdict

    groups: dict[Tuple[Path, str], List[DatasetItem]] = defaultdict(list)
    for it in items:
        p = Path(it.path)
        name_lower = p.name.lower()
        base = None
        for suf in ["_target.fits", "_ref.fits", "_diff.fits"]:
            if name_lower.endswith(suf):
                base = p.name[: -len(suf)]
                break
        if base is None:
            continue
        groups[(p.parent, base)].append(it)

    keys = list(groups.keys())[: args.limit]

    single_timings: List[float] = []
    triplet_timings: List[float] = []

    for parent, base in keys:
        target = parent / f"{base}_target.fits"
        ref = parent / f"{base}_ref.fits"
        diff = parent / f"{base}_diff.fits"

        for _ in range(args.warmups):
            for p in [target, ref, diff]:
                if p.exists():
                    _ = safe_open_image(p)
            _ = compose_triplet(target, ref, diff, size=args.size)

        for p in [target, ref, diff]:
            if not p.exists():
                continue
            ts = time_fn(lambda: safe_open_image(p), repeats=args.repeats)
            single_timings.extend(ts)

        tt = time_fn(lambda: compose_triplet(target, ref, diff, size=args.size), repeats=args.repeats)
        triplet_timings.extend(tt)

    summarize("single_from_fits", single_timings)
    summarize("triplet_compose", triplet_timings)

    if args.grid and len(keys) > 0:
        triplets: List[Tuple[Path, Path, Path]] = []
        for parent, base in keys:
            triplets.append((parent / f"{base}_target.fits", parent / f"{base}_ref.fits", parent / f"{base}_diff.fits"))
        triplets = triplets[: args.grid]

        def task(tup: Tuple[Path, Path, Path]):
            t, r, d = tup
            _ = compose_triplet(t, r, d, size=args.size)

        t0 = time.perf_counter()
        with futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            list(ex.map(task, triplets))
        wall_ms = (time.perf_counter() - t0) * 1000.0
        per_item = wall_ms / max(1, len(triplets))
        print(
            f"grid_compose: n={len(triplets)} | concurrency={args.concurrency} | wall={wall_ms:.1f} ms | per_item={per_item:.1f} ms"
        )


if __name__ == "__main__":
    main()


