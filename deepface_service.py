import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['OMP_NUM_THREADS'] = '1'  # Prevent memory leaks

from flask import Flask, request, jsonify
from deepface import DeepFace
import requests
from io import BytesIO
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize ONNX model at startup (saves 2-3s per request)
try:
    logger.info("Initializing DeepFace model...")
    DeepFace.build_model("Facenet512", backend="onnxruntime")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Model initialization failed: {str(e)}")
    raise

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Validate input
        if not request.json or 'image_url' not in request.json:
            return jsonify({"error": "Missing image_url"}), 400
            
        # Download image with timeout
        try:
            response = requests.get(
                request.json['image_url'], 
                timeout=10,
                headers={'User-Agent': 'DeepFace/1.0'}
            )
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        except Exception as e:
            return jsonify({"error": f"Image download failed: {str(e)}"}), 400

        # Analyze with optimized settings
        results = DeepFace.analyze(
            img_path=img,
            actions=['emotion'],
            detector_backend='retinaface',  # Faster than default
            enforce_detection=False,
            silent=True
        )
        
        return jsonify({
            'dominant_emotion': results[0]['dominant_emotion'],
            'emotions': results[0]['emotion']
        })
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
