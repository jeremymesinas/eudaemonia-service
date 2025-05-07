import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['OMP_NUM_THREADS'] = '1'       # Prevent memory leaks
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from flask import Flask, request, jsonify
from deepface import DeepFace
import requests
from io import BytesIO
from PIL import Image
import logging
import numpy as np

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize model at startup
logger.info("Initializing DeepFace model...")
try:
    model = DeepFace.build_model("Facenet512")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Model initialization failed: {str(e)}")
    raise

@app.route('/analyze', methods=['POST'])
import numpy as np  # Add this at the top of your file

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if not request.json or 'image_url' not in request.json:
            return jsonify({"error": "Missing image_url"}), 400
            
        response = requests.get(request.json['image_url'], timeout=10)
        img = Image.open(BytesIO(response.content))
        
        # Convert PIL Image to NumPy array (RGB format)
        img_np = np.array(img.convert('RGB'))  # <<< FIX HERE
        
        results = DeepFace.analyze(
            img_path=img_np,  # Now passing a NumPy array
            actions=['emotion'],
            detector_backend='retinaface',
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
