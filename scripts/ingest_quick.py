import time
from pathlib import Path
import sys

sys.path.insert(0, '.')

from backend import ingest_worker
from backend.db import engine
from sqlalchemy import inspect

def main():
    data_dir = Path('ATLAS_TRANSIENTS').resolve()
    exts = {'.fits', '.fit', '.fits.fz', '.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    print('starting ingest...', data_dir)
    r = ingest_worker.start(
        data_dir=data_dir,
        extensions=exts,
        by_groups=True,
        max_groups=50,
        make_thumbs=False,
        require_all_roles=None,
        batch_size=200,
        skip_hash=False,
        backfill_hashes=False,
    )
    print('start result:', r)
    for _ in range(120):
        st = ingest_worker.status()
        print('status:', st.get('message'), st.get('done'), '/', st.get('total'), 'ingested:', st.get('ingested'), 'errors:', st.get('errors'))
        if not st.get('running'):
            break
        time.sleep(0.5)

    from sqlmodel import Session, select
    from backend.db import DatasetItem
    with Session(engine) as s:
        cnt = len(s.exec(select(DatasetItem)).all())
        print('DatasetItem count:', cnt)

if __name__ == '__main__':
    main()


