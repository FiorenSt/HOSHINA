#!/usr/bin/env python3
"""Show all HDUs in FITS files to identify the correct one."""

import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize, LinearStretch

def zscale_image(data, contrast=0.1):
    """Apply ZScale normalization with specified contrast."""
    try:
        # Clean data
        x = np.array(data, dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # ZScale with custom contrast
        interval = ZScaleInterval(contrast=contrast)
        lo, hi = interval.get_limits(x)
        
        if hi > lo:
            y = (x - lo) / (hi - lo)
            y = np.clip(y, 0.0, 1.0)
        else:
            y = np.zeros_like(x)
        
        # Convert to 8-bit
        img8 = (y * 255.0).astype(np.uint8)
        return img8
    except Exception as e:
        print(f"ZScale failed: {e}")
        # Fallback to simple scaling
        x = np.array(data, dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        p2, p98 = np.percentile(x, [2, 98])
        if p98 > p2:
            y = (x - p2) / (p98 - p2)
            y = np.clip(y, 0.0, 1.0)
        else:
            y = np.zeros_like(x)
        return (y * 255.0).astype(np.uint8)

def create_hdu_grid(hdus, filename, output_dir):
    """Create a grid showing all HDUs with data."""
    # Find HDUs with 2D data
    valid_hdus = []
    for i, hdu in enumerate(hdus):
        if hasattr(hdu, 'data') and hdu.data is not None and hdu.data.ndim >= 2:
            data = hdu.data
            if data.ndim > 2:
                data = data[0]  # Take first slice
            valid_hdus.append((i, hdu.name or f"HDU{i}", data))
    
    if not valid_hdus:
        print(f"No 2D data found in {filename}")
        return
    
    # Create grid layout
    cols = min(4, len(valid_hdus))
    rows = (len(valid_hdus) + cols - 1) // cols
    
    # Image size per HDU
    hdu_size = 200
    spacing = 20
    title_height = 30
    
    total_w = cols * hdu_size + (cols - 1) * spacing
    total_h = rows * (hdu_size + title_height) + (rows - 1) * spacing
    
    canvas = Image.new("RGB", (total_w, total_h), color=(16, 16, 16))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    
    for idx, (hdu_idx, name, data) in enumerate(valid_hdus):
        row = idx // cols
        col = idx % cols
        
        x = col * (hdu_size + spacing)
        y = row * (hdu_size + title_height + spacing)
        
        # Process image
        try:
            img8 = zscale_image(data, contrast=0.1)
            if img8.ndim == 2:
                rgb = np.stack([img8] * 3, axis=-1)
            else:
                rgb = img8
            
            # Resize to fit
            pil_img = Image.fromarray(rgb.astype(np.uint8))
            pil_img = pil_img.resize((hdu_size, hdu_size), Image.BICUBIC)
            
            # Paste image
            canvas.paste(pil_img, (x, y + title_height))
            
            # Add label
            label = f"{name} ({data.shape})"
            draw.text((x, y), label, fill=(255, 255, 255), font=font)
            
            # Add stats
            stats = f"min:{np.min(data):.0f} max:{np.max(data):.0f}"
            draw.text((x, y + title_height - 15), stats, fill=(200, 200, 200), font=font)
            
        except Exception as e:
            # Error placeholder
            error_img = Image.new("RGB", (hdu_size, hdu_size), color=(64, 0, 0))
            canvas.paste(error_img, (x, y + title_height))
            draw.text((x, y), f"{name} ERROR", fill=(255, 0, 0), font=font)
            draw.text((x, y + title_height - 15), str(e)[:30], fill=(255, 100, 100), font=font)
    
    # Save
    output_path = output_dir / f"{filename.stem}_all_hdus.png"
    canvas.save(output_path)
    print(f"Saved: {output_path}")

def main():
    data_dir = Path("ATLAS_TRANSIENTS")
    output_dir = Path("store")
    output_dir.mkdir(exist_ok=True)
    
    # Process a few sample files
    sample_files = list(data_dir.glob("*_target.fits"))[:3]
    
    for fits_file in sample_files:
        print(f"\nProcessing {fits_file.name}...")
        try:
            with fits.open(fits_file) as hdul:
                print(f"  HDUs: {len(hdul)}")
                for i, hdu in enumerate(hdul):
                    name = getattr(hdu, 'name', f'HDU{i}')
                    data = getattr(hdu, 'data', None)
                    if data is not None:
                        print(f"    {i}: {name} - shape: {data.shape}, dtype: {data.dtype}")
                        if data.ndim >= 2:
                            print(f"        min: {np.nanmin(data):.1f}, max: {np.nanmax(data):.1f}, median: {np.nanmedian(data):.1f}")
                
                create_hdu_grid(hdul, fits_file, output_dir)
                
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    main()
