import os
import gc
import logging
import traceback
from functools import lru_cache
from io import BytesIO
import numpy as np
import requests
from PIL import Image, UnidentifiedImageError
from flask import Flask, request, jsonify
from deepface import DeepFace

# ========== Environment Configuration ==========
os.environ.update({
    'TF_CPP_MIN_LOG_LEVEL': '3',          # Suppress TensorFlow logs
    'OMP_NUM_THREADS': '1',               # Prevent memory leaks
    'CUDA_VISIBLE_DEVICES': '',            # Disable GPU
    'TF_FORCE_GPU_ALLOW_GROWTH': 'true'   # Better memory handling
})

# ========== Flask Application Setup ==========
app = Flask(__name__)

# ========== Logging Configuration ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========== Model Loading with Cache ==========
@lru_cache(maxsize=1)
def get_face_model():
    """Load and cache the face analysis model"""
    try:
        logger.info("Loading FaceNet model...")
        return DeepFace.build_model("Facenet")
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}\n{traceback.format_exc()}")
        raise RuntimeError("Could not load face analysis model")

# ========== Image Processing Utilities ==========
def download_image(url, timeout=15):
    """Download image with robust error handling"""
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={'User-Agent': 'Mozilla/5.0'},
            stream=True
        )
        response.raise_for_status()
        
        # Verify content type
        if 'image/' not in response.headers.get('Content-Type', '').lower():
            raise ValueError("URL does not point to an image")
            
        # Check image size (max 2MB for free tier)
        content_length = int(response.headers.get('Content-Length', 0))
        if content_length > 2 * 1024 * 1024:
            raise ValueError("Image exceeds maximum size (2MB)")
            
        return BytesIO(response.content)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed: {str(e)}")
        raise ValueError("Failed to download image")
    except Exception as e:
        logger.error(f"Image validation failed: {str(e)}")
        raise

def process_image(image_data):
    """Convert image to numpy array with validation"""
    try:
        with Image.open(image_data) as img:
            # Verify supported format
            if img.format not in ['JPEG', 'PNG', 'WEBP']:
                raise ValueError(f"Unsupported format: {img.format}")
                
            # Convert to RGB numpy array
            img_np = np.array(img.convert('RGB'))
            
            # Optional: Resize large images to save memory
            if max(img_np.shape) > 1024:
                img_np = np.array(img.resize((512, 512)).convert('RGB'))
                
            return img_np
    except UnidentifiedImageError:
        raise ValueError("Invalid or corrupt image file")
    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}")
        raise ValueError("Could not process image")

# ========== API Endpoints ==========
@app.route('/analyze', methods=['POST'])
def analyze_face():
    """Main face analysis endpoint"""
    try:
        # Validate input
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
            
        data = request.get_json()
        if not data or 'image_url' not in data:
            return jsonify({"error": "Missing image_url parameter"}), 400
            
        # Download and process image
        try:
            image_data = download_image(data['image_url'])
            img_array = process_image(image_data)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
            
        # Analyze face with memory safeguards
        try:
            results = DeepFace.analyze(
                img_path=img_array,
                actions=['emotion'],
                detector_backend='opencv',  # Lightweight detector
                enforce_detection=False,
                silent=True
            )
            
            if not results:
                return jsonify({"error": "No faces detected"}), 400
                
            # Clean up memory immediately
            del img_array
            gc.collect()
            
            return jsonify({
                'dominant_emotion': results[0]['dominant_emotion'],
                'emotions': results[0]['emotion']
            })
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}\n{traceback.format_exc()}")
            return jsonify({"error": "Face analysis failed"}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Service health endpoint"""
    try:
        # Test model availability
        get_face_model()
        return jsonify({
            "status": "healthy",
            "memory_usage": f"{os.getpid()} - {gc.mem_free() / 1024 / 1024:.2f}MB free"
        })
    except Exception:
        return jsonify({"status": "unhealthy"}), 500

# ========== Main Execution ==========
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
