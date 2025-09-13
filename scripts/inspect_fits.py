#!/usr/bin/env python3
"""Quick FITS inspector to understand the data structure."""

import sys
from pathlib import Path

# Add project root to path
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parent.parent
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
                    if hasattr(hdu, 'header'):
                        # Show some key header info
                        keys = ['NAXIS', 'NAXIS1', 'NAXIS2', 'BITPIX', 'BZERO', 'BSCALE']
                        for key in keys:
                            if key in hdu.header:
                                print(f"      {key}: {hdu.header[key]}")
                else:
                    print(f"  {i}: {name} - no data")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Check a few sample files
    data_dir = Path("ATLAS_TRANSIENTS")
    sample_files = list(data_dir.glob("*_target.fits"))[:3]
    
    for f in sample_files:
        inspect_fits(f)
        print()
