import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logs
os.environ['OMP_NUM_THREADS'] = '1'       # Prevent memory leaks
os.environ['CUDA_VISIBLE_DEVICES'] = ''   # Disable GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Better memory handling

from flask import Flask, request, jsonify
from deepface import DeepFace
import requests
from io import BytesIO
from PIL import Image
import logging
import numpy as np
from functools import lru_cache
import gc

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Cache the model with memory cleanup
@lru_cache(maxsize=1)
def load_model():
    logger.info("Initializing DeepFace model...")
    try:
        model = DeepFace.build_model("Facenet")  # Using lighter Facenet instead of Facenet512
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        raise

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Fail fast validation
        if not request.json or 'image_url' not in request.json:
            return jsonify({"error": "Missing image_url"}), 400
            
        # Stream image download to save memory
        with requests.get(request.json['image_url'], stream=True, timeout=15) as r:
            r.raise_for_status()
            img = Image.open(BytesIO(r.content))
        
        # Convert and immediately free memory
        img_np = np.array(img.convert('RGB'))
        del img  # Explicit memory release
        gc.collect()  # Force garbage collection
        
        # Use faster detector and single action
        results = DeepFace.analyze(
            img_path=img_np,
            actions=['emotion'],  # Only emotion to reduce processing
            detector_backend='fastmtcnn',  # Lighter than retinaface
            enforce_detection=False,
            silent=True
        )
        
        # Clean up before response
        del img_np
        gc.collect()
        
        return jsonify({
            'dominant_emotion': results[0]['dominant_emotion'],
            'emotions': results[0]['emotion']
        })
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Image download failed: {str(e)}")
        return jsonify({"error": "Failed to download image"}), 400
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return jsonify({"error": "Processing error. Please try a different image."}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

# Memory cleanup between requests
@app.teardown_request
def cleanup(exception=None):
    gc.collect()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
