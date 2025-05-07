import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from flask import Flask, request, jsonify
from deepface import DeepFace
import requests
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import logging
import numpy as np
from functools import lru_cache
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@lru_cache(maxsize=1)
def load_model():
    try:
        logger.info("Loading Facenet model...")
        return DeepFace.build_model("Facenet")
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}\n{traceback.format_exc()}")
        raise

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Validate input
        if not request.json:
            return jsonify({"error": "No JSON payload provided"}), 400
        if 'image_url' not in request.json:
            return jsonify({"error": "Missing image_url parameter"}), 400
        
        # Download image with error handling
        try:
            response = requests.get(
                request.json['image_url'], 
                timeout=15,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            response.raise_for_status()
            
            # Verify image content
            if 'image/' not in response.headers.get('Content-Type', ''):
                return jsonify({"error": "URL does not point to an image"}), 400
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Image download failed: {str(e)}")
            return jsonify({"error": "Failed to download image. Please check the URL."}), 400

        # Process image with detailed error handling
        try:
            with Image.open(BytesIO(response.content)) as img:
                if img.format not in ['JPEG', 'PNG', 'WEBP']:
                    return jsonify({"error": f"Unsupported image format: {img.format}"}), 400
                
                img_np = np.array(img.convert('RGB'))
                
                # Analyze with specific error capture
                try:
                    results = DeepFace.analyze(
                        img_path=img_np,
                        actions=['emotion'],
                        detector_backend='fastmtcnn',
                        enforce_detection=False,
                        silent=False  # Set to False to get detection logs
                    )
                    
                    if not results:
                        return jsonify({"error": "No faces detected"}), 400
                        
                    return jsonify({
                        'dominant_emotion': results[0]['dominant_emotion'],
                        'emotions': results[0]['emotion']
                    })
                    
                except Exception as e:
                    logger.error(f"Analysis failed: {str(e)}\n{traceback.format_exc()}")
                    return jsonify({"error": f"Analysis error: {str(e)}"}), 500
                    
        except UnidentifiedImageError:
            return jsonify({"error": "Invalid image file"}), 400
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}\n{traceback.format_exc()}")
            return jsonify({"error": "Image processing error"}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health():
    try:
        # Test model loading if not already loaded
        load_model()
        return jsonify({"status": "healthy"})
    except:
        return jsonify({"status": "unhealthy"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
