# Quick Start Guide

## 🚀 Getting Started

### 1. Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# Optional: Install astropy for FITS support
pip install astropy

# Optional: Install PyTorch for better embeddings
pip install torch torchvision
```

### 2. Set Up Your Dataset
```bash
# Set the data directory environment variable (point to your images folder)
set AL_DATA_DIR=C:\Users\fiore\Desktop\active_labeler_app\ATLAS_TRANSIENTS

# Ingest your images (supports JPG, PNG, TIFF, FITS)
python -m backend.cli.ingest --data-dir %AL_DATA_DIR%

# Set up default classes (you can customize these later)
python -m backend.cli.setup_classes
```

### 3. Run the Application
```bash
# Start the server
uvicorn backend.main:app --reload --port 8000

# Open your browser to http://localhost:8000
```

## 🎯 Labeling Your Images

### Default Classes
- **1** - Class_1 (customize in UI)
- **2** - Class_2 (customize in UI)
- **3** - Class_3 (customize in UI)
- **0** - Unsure (need more information)
- **X** - Skip (not applicable)

### Customizing Classes
Click the **"Classes"** button in the UI to:
- Rename classes to match your data
- Add more classes
- Remove classes you don't need
- Set keyboard shortcuts

### Workflow
1. **Start with Uncertain queue** - Shows items the model is least confident about
2. **Use keyboard shortcuts** - Press 1 or 2 to quickly label
3. **Batch operations** - Select multiple images with checkboxes, then label all at once
4. **Retrain frequently** - Press R to retrain the model with your labels
5. **Use Similar search** - Click "Similar" to find related images for batch labeling

### Tips for Efficient Labeling
- Use the **Map** view to see clustering patterns in your data
- Check **Stats** to monitor your labeling progress
- Start with a few labeled examples, then retrain the model
- Use **Similar** search to find related images for batch labeling
- The **Uncertain** queue shows items the model is least confident about

## 🔧 Advanced Features

### Active Learning Queues
- **Uncertain**: Items the model is least confident about
- **Diverse**: Representative samples across the dataset
- **Oddities**: Outliers that might be interesting edge cases
- **Band**: Items with specific probability ranges

### Batch Operations
- Select multiple images with checkboxes
- Use Ctrl+A to select all visible items
- Press Escape to clear selection
- Batch label with keyboard shortcuts

### Keyboard Shortcuts
- **1-9**: Label with class (or batch label if items selected)
- **0**: Mark as unsure
- **X**: Skip item
- **R**: Retrain model
- **Ctrl+A**: Select all visible items
- **Escape**: Clear selection
- **H** or **?**: Show help
- **Click images**: Zoom to full size

## 📊 Monitoring Progress

The app shows:
- **Progress bar**: Overall labeling completion percentage
- **Stats**: Detailed breakdown of labeled vs unlabeled items
- **Class distribution**: How many items per class
- **Queue information**: Current active learning strategy

## 🚨 Troubleshooting

### Common Issues
1. **Import errors**: Make sure you're in the correct directory
2. **FITS files not loading**: Install astropy with `pip install astropy`
3. **Slow performance**: Reduce page size or install PyTorch for faster embeddings
4. **Database errors**: Delete `store/app.db` to reset

### Performance Tips
- Use smaller page sizes for large datasets
- Install PyTorch for faster ResNet18 embeddings
- Close unused browser tabs
- Use batch operations for efficiency

## 📁 File Structure
```
ATLAS_TRANSIENTS/
├── [your FITS files]    # All your images in one folder
└── [more images...]     # Supports JPG, PNG, TIFF, FITS
```

The app will recursively scan your folder and process all supported image formats.
