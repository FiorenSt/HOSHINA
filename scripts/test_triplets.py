#!/usr/bin/env python3
"""Test triplet finding logic."""

import sys
from pathlib import Path

# Add project root to path
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Demo script removed. Keep this test as a placeholder or implement local logic.
def find_triplets(data_dir: Path):
    return []

def main():
    data_dir = Path("ATLAS_TRANSIENTS")
    trips = find_triplets(data_dir)
    
    print(f"Found {len(trips)} triplets")
    
    for i, t in enumerate(trips[:5]):
        print(f"{i}: {t['base']}")
        print(f"  target: {t['target'] is not None} ({t['target']})")
        print(f"  ref: {t['ref'] is not None} ({t['ref']})")
        print(f"  diff: {t['diff'] is not None} ({t['diff']})")
        print()

if __name__ == "__main__":
    main()
