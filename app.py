# app.py (no-save, no-chart)
import os
import time
import uuid
import io
import base64
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
from threading import Lock
import json

# tensorflow / keras
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_preprocess

# ---------- CONFIG ----------
ALLOWED_EXT = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB

LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ---------- APP ----------
app = Flask(__name__)
app.secret_key = 'replace-with-a-secure-random-key'
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# ---------- LOAD MODELS ----------
# adjust paths to your model files
RESNET_PATH = 'models/resnet50v2.h5'
VGG_PATH = 'models/vgg19v2.h5'

print("Loading models...")
try:
    model_resnet = load_model(RESNET_PATH)
    model_vgg = load_model(VGG_PATH)
except Exception as e:
    print("Error loading models:", e)
    raise
print("Models loaded.")

model_lock = Lock()

# ---------- Helpers ----------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def pil_to_base64(pil_img, fmt='JPEG', quality=85):
    """
    Convert PIL image to data URI base64 (for embedding in <img src="data:...">).
    Returns the data URI string.
    """
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt, quality=quality)
    byte_im = buf.getvalue()
    b64 = base64.b64encode(byte_im).decode('utf-8')
    return f"data:image/{fmt.lower()};base64,{b64}"

def preprocess_image_for_model(pil_img, model_name='resnet50', target_size=(224,224)):
    """
    Resize + preprocess for model. Returns np array (1,H,W,3).
    pil_img: PIL.Image (RGB or L)
    """
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    img_resized = pil_img.resize(target_size)
    arr = np.array(img_resized).astype('float32')  # 0-255
    if model_name == 'resnet50':
        x = resnet_preprocess(arr)   # expects 0-255
    elif model_name == 'vgg19':
        x = vgg_preprocess(arr)      # expects 0-255
    else:
        x = arr / 255.0
    x = np.expand_dims(x, axis=0)
    return x

def predict_and_time(model, preprocessed_img):
    t0 = time.time()
    with model_lock:
        preds = model.predict(preprocessed_img)
    t1 = time.time()
    elapsed = t1 - t0
    probs = np.asarray(preds[0], dtype=float)
    if probs.sum() == 0 or probs.max() > 1.1 or probs.min() < -1.0:
        e = np.exp(probs - np.max(probs))
        probs = e / e.sum()
    probs = probs / probs.sum()
    return probs, elapsed

# ---------- Routes ----------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image')
        if file is None or file.filename == '':
            flash('No file uploaded')
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash('File type not allowed. Allowed: png, jpg, jpeg')
            return redirect(request.url)

        model_choice = request.form.get('model')
        if model_choice not in ('resnet50', 'vgg19'):
            flash('Choose a model')
            return redirect(request.url)

        # Read image from file stream (do NOT save to disk)
        try:
            pil_img = Image.open(file.stream).convert('RGB')
        except Exception as e:
            flash('Cannot open image file.')
            return redirect(request.url)

        # Keep a copy for display (optionally resize for preview)
        preview_img = pil_img.copy()
        # optional: limit preview size to keep payload small
        preview_img.thumbnail((800, 800))

        # Preprocess for model (use a separate copy)
        if model_choice == 'resnet50':
            x = preprocess_image_for_model(pil_img, model_name='resnet50')
            probs, elapsed = predict_and_time(model_resnet, x)
        else:
            x = preprocess_image_for_model(pil_img, model_name='vgg19')
            probs, elapsed = predict_and_time(model_vgg, x)

        # Prepare display data
        image_data_uri = pil_to_base64(preview_img, fmt='JPEG', quality=85)
        top_idx = probs.argsort()[::-1]
        results = [(LABELS[i], float(probs[i])) for i in top_idx]
        time_str = f"{elapsed*1000:.0f} ms" if elapsed < 1 else f"{elapsed:.3f} s"

        # Pass to template (probs and labels as JSON for potential JS usage)
        return render_template(
            'result.html',
            image_data=image_data_uri,
            results=results,
            time_elapsed=time_str,
            model_name=model_choice,
            probs=json.dumps([float(p) for p in probs]),
            labels=json.dumps(LABELS)
        )

    return render_template('index.html')

# optional cleanup or health routes can be added

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
