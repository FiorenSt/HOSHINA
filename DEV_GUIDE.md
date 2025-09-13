## Development Guide (Windows PowerShell)

This guide shows how to run HOSHINA for development on Windows using PowerShell. Commands are separated with semicolons (;), not &&.

### 1) Prerequisites
- Python 3.9+
- PowerShell
- Optional (recommended for performance/features):
  - Astropy for FITS preview: `pip install astropy`
  - TensorFlow CPU (Windows AMD64): `pip install tensorflow-cpu`

### 2) Clone and enter project
```powershell
cd C:\Users\fiore\Desktop\active_labeler_app
```

### 3) Create virtual environment and install dependencies
```powershell
python -m venv venv; .\venv\Scripts\Activate.ps1;
pip install --upgrade pip; pip install -r requirements.txt;
# Optional extras:
# pip install astropy;
```

### 4) Set data directory and ingest images
Point `AL_DATA_DIR` to a folder with your images (JPG/PNG/TIFF/FITS). The app does not copy files; it references them by path.
```powershell
$env:AL_DATA_DIR = "C:\Users\fiore\Desktop\active_labeler_app\ATLAS_TRANSIENTS";
python .\scripts\ingest.py --data-dir $env:AL_DATA_DIR;
python .\scripts\setup_classes.py;
```

### 5) Run the development server
```powershell
uvicorn backend.main:app --reload --port 8000;
```

Open `http://localhost:8000` in your browser. Health check: `http://localhost:8000/healthz`.

### 6) Using the UI
- Queue: choose Uncertain/Diverse/Oddities/Band.
- Page size: adjust with the Page size selector in the header.
- Keyboard: 1-9 label, 0 unsure, X skip, R retrain, Ctrl+A select all, Esc clear.
- Batch: select multiple tiles (checkbox) and apply unsure/skip or number keys.
- Retrain: re-fit the assistive model on your labeled items.
- Map: compute or load cached UMAP embedding overview (install `umap-learn`).
- Export: download `labels.csv`, `classes.json`, and `manifest.json`.

### 7) Troubleshooting (Windows)
- If you see SQLite pool timeouts, they are addressed in code via better session scoping and pool settings. Restart the server after updating.
- FITS preview errors: `pip install astropy`.
- Slow embeddings: install PyTorch for ResNet18 encoder; otherwise HOG is used.
- Reset DB (dangerous): delete `store\\app.db` (you will lose labels/classes).

### 8) Developer tips
- Code lives under `backend/` and `frontend/`. Static assets are served from `frontend/assets/`.
- Thumbnails and models are stored in `store/`.
- Environment overrides: `AL_STORE_DIR`, `AL_DB_PATH`, `AL_EMBEDDING_BACKEND` (`auto|torch|hog`).


