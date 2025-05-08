import os
import gc
import logging
from functools import lru_cache
from io import BytesIO
import numpy as np
import requests
from PIL import Image, UnidentifiedImageError
from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
from deepface.commons import functions

# ========== Configuration ==========
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure environment for optimal performance
os.environ.update({
    'TF_CPP_MIN_LOG_LEVEL': '3',      # Suppress TensorFlow logs
    'OMP_NUM_THREADS': '1',           # Prevent memory leaks
    'TF_FORCE_GPU_ALLOW_GROWTH': 'true'
})

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========== Model Management ==========
@lru_cache(maxsize=1)
def load_face_model():
    """Cache the face analysis model to improve performance"""
    try:
        logger.info("Loading FaceNet model...")
        return DeepFace.build_model("Facenet")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise

# Pre-load the model when starting up
try:
    face_model = load_face_model()
except Exception as e:
    logger.critical(f"Failed to initialize model: {e}")
    exit(1)

# ========== Image Processing ==========
def validate_and_process_image(image_url):
    """
    Download and validate an image from URL
    Returns processed numpy array or raises ValueError
    """
    try:
        # Download image with timeout
        response = requests.get(
            image_url,
            timeout=15,
            headers={'User-Agent': 'DeepFaceAPI/1.0'},
            stream=True
        )
        response.raise_for_status()

        # Validate content type and size
        content_type = response.headers.get('Content-Type', '').lower()
        if not content_type.startswith('image/'):
            raise ValueError("URL does not point to an image")

        # Process image
        with Image.open(BytesIO(response.content)) as img:
            if img.format not in ['JPEG', 'PNG', 'WEBP']:
                raise ValueError(f"Unsupported image format: {img.format}")

            # Convert to RGB and resize if too large
            img_array = np.array(img.convert('RGB'))
            if max(img_array.shape) > 1024:
                img_array = np.array(img.resize((512, 512)).convert('RGB'))
                
            return img_array

    except requests.exceptions.RequestException as e:
        raise ValueError(f"Image download failed: {e}")
    except UnidentifiedImageError:
        raise ValueError("Invalid or corrupt image file")
    except Exception as e:
        raise ValueError(f"Image processing error: {e}")

# ========== API Endpoints ==========
@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_face():
    """
    Analyze facial emotion from image URL
    Expected JSON: {"image_url": "https://example.com/image.jpg"}
    """
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200

    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()
        image_url = data.get('image_url')
        
        if not image_url:
            return jsonify({"error": "Missing image_url parameter"}), 400

        # Process image
        try:
            img_array = validate_and_process_image(image_url)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        # Analyze face
        try:
            results = DeepFace.analyze(
                img_path=img_array,
                actions=['emotion'],
                detector_backend='retinaface',
                enforce_detection=False,
                silent=True,
                model=face_model  # Use our pre-loaded model
            )

            if not results:
                return jsonify({"error": "No faces detected"}), 400

            # Format response for Flutter
            return jsonify({
                "status": "success",
                "dominant_emotion": results[0]['dominant_emotion'],
                "emotion_scores": results[0]['emotion'],
                "face_region": results[0]['region'] if 'region' in results[0] else None
            })

        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return jsonify({"error": "Face analysis failed"}), 500

        finally:
            # Clean up memory
            del img_array
            gc.collect()

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    try:
        return jsonify({
            "status": "healthy",
            "model": "loaded" if face_model else "unavailable"
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

# ========== Main Execution ==========
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        threaded=True,
        debug=False
    )
