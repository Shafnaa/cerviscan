from flask import Flask, redirect, url_for, render_template, request, session, jsonify, send_file
from werkzeug.utils import secure_filename
import numpy as np
import os
import cv2
from PIL import Image
import io
import base64

# Direktori untuk menyimpan file gambar
UPLOAD_FOLDER = 'static/image_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = "RahasiaTau"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Fungsi untuk memproses gambar (cropping, grayscale, segmentasi)
def process_image(file):
    # Membaca file sebagai numpy array
    image = Image.open(file)
    image = image.convert("RGB")
    np_image = np.array(image)

    # Crop bagian tengah gambar (contoh sederhana)
    h, w, _ = np_image.shape
    cropped = np_image[h // 4: 3 * h // 4, w // 4: 3 * w // 4]

    # Convert ke grayscale
    grayscale = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)

    # Segmentasi sederhana (thresholding)
    _, segmented = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY)

    return cropped, grayscale, segmented

# Konversi gambar ke Base64 untuk ditampilkan di HTML
def convert_to_base64(image, mode="RGB"):
    img_pil = Image.fromarray(image, mode)
    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file:
        # Proses gambar
        cropped, grayscale, segmented = process_image(file)

        # Konversi hasil ke Base64
        cropped_base64 = convert_to_base64(cropped)
        grayscale_base64 = convert_to_base64(grayscale, mode="L")
        segmented_base64 = convert_to_base64(segmented, mode="L")

        # Kirim hasil ke frontend
        response = {
            "cropped_image": f"data:image/png;base64,{cropped_base64}",
            "grayscale_image": f"data:image/png;base64,{grayscale_base64}",
            "segmented_image": f"data:image/png;base64,{segmented_base64}",
            "features": {
                "feature1": 0.75,
                "feature2": 1.23,
                "feature3": 0.89,
                "feature4": 1.05,
                "feature5": 0.67
            },
            "result": "Image Status: Positive"
        }
        return jsonify(response)

    return jsonify({"error": "Invalid file"}), 400

if __name__ == '__main__':
    app.run(debug=True)
