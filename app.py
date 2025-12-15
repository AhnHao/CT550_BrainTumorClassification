# app.py (no-save, no-chart)
import os
import time
import uuid
import io
import base64
from flask import Flask, request, render_template, redirect, url_for, flash, send_file, session
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
from threading import Lock
import json
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from report_template import generate_report_pdf  # <-- import hàm sinh báo cáo

# tensorflow / keras
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_preprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pathlib

# ---------- CONFIG ----------
ALLOWED_EXT = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB

LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']

# create tmp directory for images
TMP_DIR = os.path.join(os.path.dirname(__file__), "tmp")
os.makedirs(TMP_DIR, exist_ok=True)

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

def generate_gradcam(model, img_array, orig_img, class_idx=None, layer_name=None):
    """
    Sinh ảnh Grad-CAM cho mô hình Keras.
    img_array: ảnh đã preprocess (1,224,224,3)
    orig_img: PIL.Image gốc (RGB)
    class_idx: chỉ số lớp cần giải thích (nếu None sẽ lấy lớp dự đoán cao nhất)
    layer_name: tên lớp convolution cuối (nếu None sẽ tự động lấy)
    Trả về: PIL.Image đã chồng heatmap
    """
    import tensorflow as tf
    from tensorflow.keras import backend as K

    # Tìm lớp conv cuối nếu chưa chỉ định
    if layer_name is None:
        for layer in reversed(model.layers):
            if 'conv' in layer.name or 'bn' in layer.name:
                layer_name = layer.name
                break

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if class_idx is None:
            class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    heatmap = heatmap.numpy()

    # Resize heatmap về đúng size ảnh gốc
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize(orig_img.size, resample=Image.BILINEAR)
    heatmap = np.array(heatmap)

    # Áp dụng colormap
    import cv2
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Chồng heatmap lên ảnh gốc
    superimposed_img = np.array(orig_img)
    # Đảm bảo cả hai đều là uint8
    if superimposed_img.dtype != np.uint8:
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    if heatmap_color.dtype != np.uint8:
        heatmap_color = np.clip(heatmap_color, 0, 255).astype(np.uint8)
    superimposed_img = cv2.addWeighted(superimposed_img, 0.6, heatmap_color, 0.4, 0)
    superimposed_img = np.uint8(superimposed_img)
    return Image.fromarray(superimposed_img)

def safe_pil_to_b64(img, fmt='JPEG', quality=85):
	"""
	Encode PIL image to base64 (raw, no data URI). On error return a small placeholder image encoded.
	"""
	try:
		buf = io.BytesIO()
		img.save(buf, format=fmt, quality=quality)
		return base64.b64encode(buf.getvalue()).decode('utf-8')
	except Exception:
		# fallback placeholder
		ph = Image.new('RGB', (224, 224), (200, 200, 200))
		buf = io.BytesIO()
		ph.save(buf, format='JPEG', quality=75)
		return base64.b64encode(buf.getvalue()).decode('utf-8')

def safe_save_pil(img, dirpath=TMP_DIR, fmt='JPEG', quality=85):
	"""Save PIL image to tmp dir, return filename (basename)."""
	fn = f"{uuid.uuid4().hex}.jpg"
	path = os.path.join(dirpath, fn)
	try:
		img.save(path, format=fmt, quality=quality)
		return fn
	except Exception:
		# fallback placeholder
		ph = Image.new('RGB', (224, 224), (200, 200, 200))
		ph.save(path, format='JPEG', quality=75)
		return fn

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
		preview_img.thumbnail((800, 800))

		# Preprocess cho grad-cam (phải resize đúng 224x224)
		gradcam_input = preprocess_image_for_model(pil_img, model_name=model_choice)

		# Sinh ảnh grad-cam thực sự (bọc try/except -> fallback về ảnh gốc nếu lỗi)
		try:
			if model_choice == 'resnet50':
				gradcam_img = generate_gradcam(model_resnet, gradcam_input, preview_img, layer_name=None)
			else:
				gradcam_img = generate_gradcam(model_vgg, gradcam_input, preview_img, layer_name=None)
		except Exception as e:
			# fallback: use resized original preview image as gradcam placeholder
			print(f"[WARN] gradcam generation failed: {e}")
			gradcam_img = preview_img.copy()

		# ensure reasonable sizes for display and for saved files
		preview_for_save = preview_img.copy()
		preview_for_save.thumbnail((800, 800))
		gradcam_for_save = gradcam_img.copy()
		gradcam_for_save.thumbnail((800, 800))

		# Save images to tmp folder and store only filenames in session
		uploaded_fname = safe_save_pil(preview_for_save)
		gradcam_fname = safe_save_pil(gradcam_for_save)

		# Debug lengths / files
		print(f"[DEBUG] saved uploaded file: {uploaded_fname}, gradcam file: {gradcam_fname}")

		# Prepare data-URI for template display (do NOT store this in session)
		image_data_uri = pil_to_base64(preview_for_save, fmt='JPEG', quality=85)
		gradcam_image_data_uri = pil_to_base64(gradcam_for_save, fmt='JPEG', quality=85)

		# Preprocess for model (use a separate copy)
		if model_choice == 'resnet50':
			x = preprocess_image_for_model(pil_img, model_name='resnet50')
			probs, elapsed = predict_and_time(model_resnet, x)
		else:
			x = preprocess_image_for_model(pil_img, model_name='vgg19')
			probs, elapsed = predict_and_time(model_vgg, x)

		# Prepare display data
		top_idx = probs.argsort()[::-1]
		results = [(LABELS[i], float(probs[i])) for i in top_idx]
		time_str = f"{elapsed*1000:.0f} ms" if elapsed < 1 else f"{elapsed:.3f} s"

		# Lưu kết quả vào session để dùng cho báo cáo PDF (store filenames only)
		session['report_data'] = {
			'model': model_choice,
			'results': results,
			'time_elapsed': time_str,
			'patient_name': 'Jane Doe',
			'doctor_name': 'Dr. Jane Smith',
			'uploaded_img_name': uploaded_fname,
			'gradcam_img_name': gradcam_fname
		}

		return render_template(
			'result.html',
			image_data=image_data_uri,
			gradcam_image_data=gradcam_image_data_uri,
			results=results,
			time_elapsed=time_str,
			model_name=model_choice,
			probs=json.dumps([float(p) for p in probs]),
			labels=json.dumps(LABELS)
		)

	return render_template('index.html')

@app.route('/report')
def report():
    report_data = session.get('report_data')
    if not report_data:
        flash('Không có dữ liệu để tạo báo cáo.')
        return redirect(url_for('index'))

    # Gọi hàm sinh báo cáo PDF từ file riêng
    buffer = generate_report_pdf(report_data)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="bao_cao_du_doan.pdf", mimetype='application/pdf')

# optional cleanup or health routes can be added

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
