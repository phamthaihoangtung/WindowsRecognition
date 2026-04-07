import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from inference_sam import WindowsRecognitor

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Directory to store processed binary masks
# PROCESSED_IMAGE_DIR = "processed_images"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#PROCESSED_IMAGE_DIR = os.path.join(BASE_DIR, "processed_images")
#UPLOADED_IMAGE_DIR = os.path.join(BASE_DIR, "uploaded_images")
PROCESSED_IMAGE_DIR = "D:/server01/processed_images"
UPLOADED_IMAGE_DIR = "D:/server01/uploaded_images"

recognitor = WindowsRecognitor("config/config_prod.yaml")

# Ensure output directory exists
os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True)
os.makedirs(UPLOADED_IMAGE_DIR, exist_ok=True)

def get_binary_mask(image_path):
    """
    Process the image and return the binary mask filename.
    """
    # Perform inference using the WindowsRecognitor
    postprocessing_mask = recognitor.recognize_from_path(image_path)

    # Save the processed mask
    mask_filename = os.path.basename(image_path).replace(".jpg", ".png").replace(".jpeg", ".png").replace(".png", ".png")
    mask_path = os.path.join(PROCESSED_IMAGE_DIR, mask_filename)
    cv2.imwrite(mask_path, (postprocessing_mask * 255).astype(np.uint8))

    return mask_filename

@app.route('/upload-image', methods=['POST'])
def upload_image():
    try:
        print("Received request!")  # Log nhận request
        print(f"Headers: {request.headers}")
        print(f"Data: {request.data.decode('utf-8')}")  # Log nội dung JSON

        # Parse JSON request payload
        request_data = request.get_json()
        if not request_data or 'path' not in request_data:
            response = {"success": False, "error": "Invalid input. Missing 'path' or 'sam_choice'."}
            print("Error:", response)
            return jsonify(response), 400

        image_path = os.path.join(UPLOADED_IMAGE_DIR, request_data['path'])

        if not os.path.exists(image_path):
            response = {"success": False, "error": f"Image file not found: {image_path}"}
            print("Error:", response)
            return jsonify(response), 404

        # Process the image
        binary_mask_path = get_binary_mask(image_path)
        # os.remove(image_path)
        response = {"success": True, "url": f"{binary_mask_path}"}
        print("Response:", response)
        return jsonify(response)

    except Exception as e:
        response = {"success": False, "error": f"Internal Server Error: {str(e)}"}
        print("Error:", response)
        return jsonify(response), 500
    
@app.route('/processed_images/<filename>')
def serve_processed_image(filename):
    """
    Serve processed binary masks as static files.
    """
    return send_from_directory(PROCESSED_IMAGE_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)


