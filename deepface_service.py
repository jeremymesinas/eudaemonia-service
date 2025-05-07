from flask import Flask, request, jsonify
from deepface import DeepFace
import requests
from io import BytesIO
from PIL import Image

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get image URL from request
        image_url = request.json['image_url']
        
        # Download image
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        
        # Analyze with DeepFace
        results = DeepFace.analyze(img_path=img, actions=['emotion'], enforce_detection=False)
        
        # Return dominant emotion
        return jsonify({
            'dominant_emotion': results[0]['dominant_emotion'],
            'emotions': results[0]['emotion']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
