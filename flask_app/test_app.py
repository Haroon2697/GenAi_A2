"""End-to-end test suite for CycleGAN Flask app using Flask test client."""
import io
import base64
import unittest
import numpy as np
from PIL import Image
from app import app


class TestCycleGANApp(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

    def create_synthetic_image(self, mode='RGB', size=(150, 150), is_sketch=False):
        """Create a synthetic in-memory image."""
        if mode == 'L':
            # Grayscale sketch-like pattern
            arr = np.random.randint(50, 200, size=size, dtype=np.uint8)
            img = Image.fromarray(arr, mode='L')
        elif mode == 'RGBA':
            arr = np.zeros((size[1], size[0], 4), dtype=np.uint8)
            arr[..., 0] = 220  # Reddish
            arr[..., 1] = 120  # Greenish
            arr[..., 2] = 80   # Blueish
            arr[..., 3] = 255  # Alpha
            img = Image.fromarray(arr, mode='RGBA')
        else:
            if is_sketch:
                # Monochromatic in RGB (R=G=B) -> saturation score = 0
                val = np.random.randint(50, 200, size=size, dtype=np.uint8)
                arr = np.stack([val, val, val], axis=-1)
            else:
                # Colorful photo with distinct R, G, B channels -> high saturation score (~40+)
                arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                arr[..., 0] = 220  # R
                arr[..., 1] = 130  # G
                arr[..., 2] = 70   # B
            img = Image.fromarray(arr, mode='RGB')

        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return buf

    def test_index_page(self):
        """Test that index page loads successfully."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'CycleGAN', response.data)

    def test_convert_photo(self):
        """Test converting a color photo -> auto-detects face, returns valid base64 PNG."""
        photo_buf = self.create_synthetic_image(mode='RGB', size=(200, 200), is_sketch=False)
        response = self.client.post('/convert', data={
            'file': (photo_buf, 'photo.png'),
            'mode': 'auto'
        }, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['detected_type'], 'face')
        self.assertEqual(data['direction'], 'face -> sketch')
        self.assertGreaterEqual(data['saturation_score'], 5.0)

        # Check valid base64 PNG output
        out_bytes = base64.b64decode(data['output_image'])
        out_img = Image.open(io.BytesIO(out_bytes))
        self.assertEqual(out_img.format, 'PNG')
        self.assertEqual(out_img.size, (128, 128))

    def test_convert_sketch(self):
        """Test converting a grayscale sketch -> auto-detects sketch, returns valid base64 PNG."""
        sketch_buf = self.create_synthetic_image(mode='L', size=(180, 180), is_sketch=True)
        response = self.client.post('/convert', data={
            'file': (sketch_buf, 'sketch.png'),
            'mode': 'auto'
        }, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['detected_type'], 'sketch')
        self.assertEqual(data['direction'], 'sketch -> face')
        self.assertLess(data['saturation_score'], 5.0)

        # Check valid base64 PNG output
        out_bytes = base64.b64decode(data['output_image'])
        out_img = Image.open(io.BytesIO(out_bytes))
        self.assertEqual(out_img.format, 'PNG')
        self.assertEqual(out_img.size, (128, 128))

    def test_convert_rgba_image(self):
        """Test converting RGBA image without crash."""
        rgba_buf = self.create_synthetic_image(mode='RGBA', size=(150, 150))
        response = self.client.post('/convert', data={
            'file': (rgba_buf, 'transparent.png'),
            'mode': 'auto'
        }, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('output_image', data)
        out_img = Image.open(io.BytesIO(base64.b64decode(data['output_image'])))
        self.assertEqual(out_img.format, 'PNG')
        self.assertEqual(out_img.size, (128, 128))

    def test_non_square_center_crop(self):
        """Test non-square image (e.g. 640x480 webcam) is center-cropped to 128x128."""
        rect_buf = self.create_synthetic_image(mode='RGB', size=(640, 480), is_sketch=False)
        response = self.client.post('/convert', data={
            'file': (rect_buf, 'webcam.jpg'),
            'mode': 'auto'
        }, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        in_img = Image.open(io.BytesIO(base64.b64decode(data['input_image'])))
        out_img = Image.open(io.BytesIO(base64.b64decode(data['output_image'])))
        self.assertEqual(in_img.size, (128, 128))
        self.assertEqual(out_img.size, (128, 128))

    def test_bad_input_no_file(self):
        """Test error when no file field is in the request."""
        response = self.client.post('/convert', data={}, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn('error', data)
        self.assertIn('No file uploaded', data['error'])

    def test_bad_input_empty_filename(self):
        """Test error when empty file selection is sent."""
        response = self.client.post('/convert', data={
            'file': (io.BytesIO(b''), '')
        }, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn('error', data)
        self.assertIn('No file selected', data['error'])

    def test_bad_input_non_image_file(self):
        """Test error when non-image text/corrupted file is uploaded."""
        fake_file = io.BytesIO(b"This is just plain text, definitely not an image!")
        response = self.client.post('/convert', data={
            'file': (fake_file, 'notes.txt')
        }, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn('error', data)

    def test_bad_input_invalid_mode(self):
        """Test error when an invalid mode string is passed."""
        photo_buf = self.create_synthetic_image(mode='RGB', size=(128, 128))
        response = self.client.post('/convert', data={
            'file': (photo_buf, 'test.png'),
            'mode': 'invalid_mode'
        }, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn('error', data)
        self.assertIn('Invalid mode', data['error'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
