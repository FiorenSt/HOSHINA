
# HOSHINA (Human-in-the-Loop Image Labeling & Active Sampling)

A lightweight, OS-agnostic web app to build high-quality training datasets for image classification.
It runs entirely in your browser backed by a Python FastAPI server, and includes:

- Keyboard-first labeling UI (grid + detail).
- Configurable classes (binary or multiclass).
- "Assistive" model: retrainable small classifier with probability *calibration*.
- Active sampling queues:
  - **Uncertain** (low margin / high entropy),
  - **Diverse** (k-center/farthest-first),
  - **Oddities** (outliers via LOF or Isolation Forest),
  - **Probability band** (e.g., 0.3–0.8) for targeted sweeps.
- Similarity search (k-NN) for sweep labeling.
- UMAP/t-SNE 2-D map overview to spot clusters and gaps.
- Reproducible export (images, labels, splits, manifest).

**No GPU required.** If TensorFlow is available, the app uses a pretrained MobileNetV2 encoder for embeddings; otherwise it falls back to lightweight HOG + color features.

### New: TensorFlow quick training

- Start a TF-backed quick train from the UI via the TF Train button. Choose backbone (MobileNetV2, EfficientNetB0, ResNet50), epochs, batch size, and optional augmentation.
- Training runs server-side and updates predictions for all items upon completion.
- Progress is visible in the modal and in the header status.

## Quickstart

1) **Install Python 3.9+** and git (or just unzip this repository).
2) Create a virtual environment and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
# Optional extras (FITS support)
pip install astropy
```

3) **Prepare your images** on the server (JPG/PNG/TIFF; optionally FITS). Note the directory path, e.g. `/data/my_images`.

4) **Ingest** your images into the SQLite database and generate thumbnails:

```bash
python scripts/ingest.py --data-dir /data/my_images
```

5) **Run the server**:

```bash
# Option A: Development
uvicorn backend.main:app --reload --port 8000

# Option B: Production-ish, Workers
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

6) **Open the UI** at http://localhost:8000 (or the server’s address).

> The server reads and serves images from your data directory by path; it does *not* copy large files by default.
> Thumbnails are cached under `store/thumbs/` and embeddings/models under `store/`.

### Image/Thumbnail Serving

- Thumbnails are served at `/api/thumb/{item_id}` and generated on demand into `store/thumbs/`.
- Full-size image preview is served at `/api/file/{item_id}`. FITS files are converted to JPEG on the fly using `astropy` + PIL.
- If FITS previews fail, install `astropy` (`pip install astropy`).

## Usage Tips

- Press number keys **1..9** to assign the corresponding class; **0** = *unsure*; **X** = *skip*.
- Use the **Queue** dropdown to switch between *Uncertain*, *Diverse*, *Oddities*, and *Band*.
- Adjust the **Band** slider (e.g., 0.3–0.8) and click **Pull** to fetch items in that probability range.
- Click **Retrain** at any time to fit/calibrate the classifier on your current labels.
- Click the **Similar** icon on a tile to fetch k-NN neighbors for sweep labeling.
- Use **Map** to see the 2-D UMAP projection; click a cluster to load items from that region.

## Configuration

- Classes are stored in the DB and editable in the UI under the **Classes** dialog.
- Environment variables:
  - `AL_DATA_DIR` (default: the path given at ingest; can be overridden)
  - `AL_DB_PATH` (default: `store/app.db`)
  - `AL_STORE_DIR` (default: `store/`)
  - `AL_EMBEDDING_BACKEND` (choices: `auto`, `torch`, `hog`; default: `auto`)
- To change image types supported during ingest: `--extensions ".jpg,.jpeg,.png,.tif,.tiff,.fits"`

## Export

Use **Export** to download a ZIP with:
- `labels.csv` (item_id, path, label, was_unsure, timestamps)
- `classes.json`
- `manifest.json` (versions, embedding backend, calibration, etc.)

## Docker (optional)

```bash
docker build -t active-labeler .
docker run --rm -p 8000:8000 -e AL_DATA_DIR=/data -v /host/images:/data active-labeler
```

## Notes on Methods (peer-reviewed grounding)

- **Active learning (uncertainty/diversity)**: Settles (2012, Synthesis Lectures); Sener & Savarese (2018, ICLR).
- **Calibration**: Guo et al. (2017, ICML) temperature scaling / Platt scaling.
- **Outliers**: Breunig et al. (2000, SIGMOD, LOF); Liu et al. (2008, ICDM, Isolation Forest).
- **Embeddings**: ResNet18 (He et al., 2016, CVPR) via Torchvision; fallback HOG (Dalal & Triggs, 2005, CVPR).
- **2-D maps**: UMAP (McInnes et al., 2018) and t-SNE (van der Maaten & Hinton, 2008, JMLR).

See `DESIGN_NOTES.md` for details and references.

## License

MIT.
