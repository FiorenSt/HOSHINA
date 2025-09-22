#!/usr/bin/env python3
"""Quick FITS inspector to understand the data structure."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astropy.io import fits
import numpy as np


def inspect_fits(path: Path):
    print(f"=== {path.name} ===")
    try:
        with fits.open(path) as hdul:
            print(f"HDUs: {len(hdul)}")
            for i, hdu in enumerate(hdul):
                name = getattr(hdu, 'name', '')
                data = getattr(hdu, 'data', None)
                if data is not None:
                    print(f"  {i}: {name} - shape: {data.shape}, dtype: {data.dtype}")
                    print(f"      min: {np.nanmin(data):.3f}, max: {np.nanmax(data):.3f}, median: {np.nanmedian(data):.3f}")
                else:
                    print(f"  {i}: {name} - no data")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    data_dir = Path("ATLAS_TRANSIENTS")
    sample_files = list(data_dir.glob("*_target.fits"))[:3]
    for f in sample_files:
        inspect_fits(f)
        print()


