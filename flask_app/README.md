# CycleGAN Face-Sketch Converter Flask App

A Flask web application for converting between face photos and sketches using CycleGAN.

## Requirements

- Python 3.8 or higher
- PyTorch 2.0.0 or higher (install according to your system: https://pytorch.org/get-started/locally/)
- Flask 3.0.0 or higher
- PIL/Pillow 10.0.0 or higher
- NumPy 1.24.0 or higher

## Installation

### Windows

1. Create a virtual environment:
```cmd
python -m venv venv
venv\Scripts\activate
```

2. Install PyTorch according to your system (visit https://pytorch.org/get-started/locally/):
```cmd
# Example for CPU-only Windows:
pip install torch torchvision torchaudio
```

3. Install remaining dependencies:
```cmd
pip install -r requirements.txt
```

3. Ensure weights are in the `weights/` folder:
   - `G_face2sketch.pth`
   - `G_sketch2face.pth`

### Linux/macOS

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install PyTorch according to your system (visit https://pytorch.org/get-started/locally/):
```bash
# Example for CPU-only Linux:
pip install torch torchvision torchaudio
```

3. Install remaining dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure weights are in the `weights/` folder:
   - `G_face2sketch.pth`
   - `G_sketch2face.pth`

## Running the App

### Windows
```cmd
python app.py
```

### Linux/macOS
```bash
python3 app.py
```

The app will start on `http://localhost:5000`

## Usage

1. Open the web interface in your browser
2. Upload an image or use the camera capture feature
3. Select mode:
   - **Auto-detect**: Automatically determines if input is a sketch or photo
   - **Force Face**: Treat input as a photo (convert to sketch)
   - **Force Sketch**: Treat input as a sketch (convert to photo)
4. Click "Convert" to see the result

## Architecture

- `models.py`: CycleGAN generator architecture (ResNet with 6 residual blocks)
- `app.py`: Flask application with image preprocessing and conversion logic
- `templates/index.html`: Web interface with file upload and camera support

## Model Details

- Image size: 128x128
- Generator: ResNet with 6 residual blocks
- Latent size: 256 channels at 32x32 resolution
- Saturation threshold: 5.0 (for auto-detection)

The app runs on CPU for compatibility. For GPU support, modify the `DEVICE` variable in `app.py`.
