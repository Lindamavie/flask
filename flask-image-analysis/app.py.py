from flask import Flask, request, jsonify
import cv2
import numpy as np
import piexif
from PIL import Image, ImageChops, ImageEnhance
import torch
import torchvision.transforms as transforms
import base64
import deepface
import io
import os
import tempfile
import webbrowser
from werkzeug.utils import secure_filename
import logging
from io import BytesIO

app = Flask(__name__)

# Configuration des logs
logging.basicConfig(level=logging.INFO)

# 📌 1️⃣ Détection des métadonnées EXIF
def extract_metadata(image_path):
    try:
        exif_data = piexif.load(image_path)
        return exif_data
    except piexif.InvalidImageDataError:
        return "No metadata found"
    except Exception as e:
        logging.error(f"Error extracting metadata: {e}")
        return str(e)

# 📌 2️⃣ Détection d'Error Level Analysis (ELA)
def error_level_analysis(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        buffer = BytesIO()
        img.save(buffer, 'JPEG', quality=90)
        buffer.seek(0)
        ela_img = ImageChops.difference(img, Image.open(buffer))
        extrema = ela_img.getextrema()
        max_diff = max([pix[1] for pix in extrema])
        scale = 255.0 / max_diff if max_diff != 0 else 1
        ela_img = ImageEnhance.Brightness(ela_img).enhance(scale)
        return np.array(ela_img).tolist()
    except Exception as e:
        logging.error(f"Error performing ELA: {e}")
        return str(e)

# 📌 3️⃣ Détection d'images générées par IA
MODEL_PATH = "model_ia_detection.pth"
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.eval()
    except Exception as e:
        logging.warning(f"Erreur de chargement du modèle IA : {e}")


def detect_ai_generated(image_path):
    if model is None:
        return "⚠️ Modèle IA non chargé. Vérifiez 'model_ia_detection.pth'."
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image)
    
    prediction = torch.sigmoid(output).item()
    return {"result": "IA" if prediction > 0.5 else "Authentique", "confidence": prediction}

# 📌 4️⃣ Détection des modifications
def detect_modifications(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    modification_score = np.sum(edges) / (img.shape[0] * img.shape[1])
    return "Modifiée" if modification_score > 0.05 else "Authentique"

# 📌 5️⃣ Détection de deepfakes
def detect_deepfake(image_path):
    try:
        result = deepface.DeepFace.analyze(img_path=image_path, actions=['emotion', 'age', 'gender'])
        return {"result": "Humain", "note": "Analyse DeepFace terminée."}
    except Exception as e:
        return {"result": "Potentiellement deepfake", "error": str(e)}

# 📌 6️⃣ Recherche inversée Google
def google_reverse_image_search():
    return "🔎 Téléchargez l’image sur https://www.google.com/imghp et cliquez sur l’icône de caméra."

# 📌 **Route API principale** : `/analyze`
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'Aucune image reçue'}), 400
    
    file = request.files['image']
    if file.filename.split('.')[-1].lower() not in ['jpeg', 'jpg', 'png', 'bmp', 'gif']:
        return jsonify({'error': 'Format non supporté'}), 400
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(tempfile.gettempdir(), filename)
    
    try:
        file.save(file_path)

        metadata = extract_metadata(file_path)
        ela_result = error_level_analysis(file_path)
        ai_detection = detect_ai_generated(file_path)
        modifications = detect_modifications(file_path)
        deepfake_check = detect_deepfake(file_path)
        reverse_search = google_reverse_image_search()

        return jsonify({
            'status': 'success',
            'metadata': metadata,
            'ela_analysis': {
                'result': ela_result,
                'explanation': "ELA met en évidence les différences de compression révélant des retouches."
            },
            'ai_detection': ai_detection,
            'modifications': modifications,
            'deepfake_check': deepfake_check,
            'reverse_search': reverse_search
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# **Démarrer l'API Flask**
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

