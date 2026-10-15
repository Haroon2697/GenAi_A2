"""Flask app for CycleGAN face-sketch conversion.
Auto-detects sketch vs photo using saturation heuristic.
"""
import os
import base64
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
import torch
from models import Generator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')

app = Flask(__name__, template_folder=TEMPLATES_DIR)

# Configuration
IMG_SIZE = 128
SKETCH_THRESHOLD = 5.0  # saturation threshold for auto-detection
DEVICE = torch.device('cpu')  # run on CPU for compatibility

# Load generators at startup
print("Loading generators...")
G_face2sketch = Generator(n_res=6, nf=64)
G_sketch2face = Generator(n_res=6, nf=64)

weights_dir = os.path.join(BASE_DIR, 'weights')
g_face_path = os.path.join(weights_dir, 'G_face2sketch.pth')
g_sketch_path = os.path.join(weights_dir, 'G_sketch2face.pth')

if not os.path.exists(g_face_path):
    raise FileNotFoundError(f"Weights not found: {g_face_path}")
if not os.path.exists(g_sketch_path):
    raise FileNotFoundError(f"Weights not found: {g_sketch_path}")

G_face2sketch.load_state_dict(torch.load(g_face_path, map_location=DEVICE, weights_only=True))
G_sketch2face.load_state_dict(torch.load(g_sketch_path, map_location=DEVICE, weights_only=True))

G_face2sketch.to(DEVICE).eval()
G_sketch2face.to(DEVICE).eval()
print("Generators loaded successfully.")


def saturation_score(img_array):
    """Calculate saturation score: mean |R-G| + mean |G-B|.
    Sketches have values near 0, photos have higher values.
    """
    img_array = img_array.astype(float)
    return float(np.abs(img_array[..., 0] - img_array[..., 1]).mean() +
                 np.abs(img_array[..., 1] - img_array[..., 2]).mean())


def preprocess_image(img_bytes):
    """Convert uploaded image bytes to tensor in [-1, 1] range.
    Handles RGBA, grayscale, non-square images (center-crop), and validates inputs.
    """
    if not img_bytes:
        raise ValueError("Image file is empty")

    try:
        img = Image.open(io.BytesIO(img_bytes))
        img.load()  # verify image decoding
    except Exception as e:
        raise ValueError("Invalid image file: could not decode image") from e

    # Handle transparency in RGBA / other modes
    if img.mode == 'RGBA':
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    # Center-crop to square before resizing to prevent distortion
    w, h = img.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    img = img.crop((left, top, left + min_dim, top + min_dim))

    # Resize to 128x128
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)

    # Convert to numpy array and normalize to [-1, 1]
    img_array = np.array(img).astype(np.float32)
    tensor = torch.from_numpy(img_array).permute(2, 0, 1) / 127.5 - 1.0
    return tensor.unsqueeze(0), img_array  # (1, 3, 128, 128) and (128, 128, 3)


def tensor_to_base64(tensor):
    """Convert tensor in [-1, 1] to base64-encoded PNG."""
    tensor = tensor.squeeze(0).permute(1, 2, 0)  # (128, 128, 3)
    img_array = ((tensor.cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def numpy_to_base64(img_array):
    """Convert numpy array in [0, 255] to base64-encoded PNG."""
    img = Image.fromarray(img_array.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/convert', methods=['POST'])
def convert():
    """Convert uploaded image: auto-detect sketch vs photo and apply appropriate generator."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        img_bytes = file.read()
        input_tensor, input_array = preprocess_image(img_bytes)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f"Failed to process image: {str(e)}"}), 400

    try:
        input_tensor = input_tensor.to(DEVICE)

        # Detect input type
        mode = request.form.get('mode', 'auto')  # auto, face, sketch
        sat_score = saturation_score(input_array)

        if mode == 'auto':
            detected_type = 'sketch' if sat_score < SKETCH_THRESHOLD else 'face'
        elif mode == 'face':
            detected_type = 'face'
        elif mode == 'sketch':
            detected_type = 'sketch'
        else:
            return jsonify({'error': 'Invalid mode'}), 400

        # Apply appropriate generator
        with torch.no_grad():
            if detected_type == 'sketch':
                # Sketch -> Photo
                output_tensor = G_sketch2face(input_tensor)
                direction = 'sketch -> face'
            else:
                # Face -> Sketch
                output_tensor = G_face2sketch(input_tensor)
                direction = 'face -> sketch'

        # Convert to base64 PNG
        input_b64 = numpy_to_base64(input_array)
        output_b64 = tensor_to_base64(output_tensor)

        return jsonify({
            'detected_type': detected_type,
            'direction': direction,
            'saturation_score': round(sat_score, 2),
            'input_image': input_b64,
            'output_image': output_b64
        })

    except Exception as e:
        return jsonify({'error': f"Inference failed: {str(e)}"}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
