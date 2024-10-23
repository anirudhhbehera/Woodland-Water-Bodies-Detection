import os
import requests
import cv2
import numpy as np
import base64
import io
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from deepforest import main
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# URLs of the models hosted on GitHub
TREE_MODEL_URL = 'https://github.com/anirudhhbehera/Woodland-Water-Bodies-Detection/raw/main/model.opendata_luftbild_dop60.patch400.ckpt'
WATER_MODEL_URL = 'https://github.com/anirudhhbehera/Woodland-Water-Bodies-Detection/raw/main/saved_model.h5'

def download_model(url, destination):
    response = requests.get(url)
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download model from {url}")

# Define paths for models
TREE_MODEL_PATH = 'model.opendata_luftbild_dop60.patch400.ckpt'
WATER_MODEL_PATH = 'saved_model.h5'

# Download the models if not already present
if not os.path.exists(TREE_MODEL_PATH):
    download_model(TREE_MODEL_URL, TREE_MODEL_PATH)

if not os.path.exists(WATER_MODEL_PATH):
    download_model(WATER_MODEL_URL, WATER_MODEL_PATH)

# Load the models
tree_model = main.deepforest.load_from_checkpoint(checkpoint_path=TREE_MODEL_PATH)
water_model = load_model(WATER_MODEL_PATH)

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        # Read the uploaded image
        image_file = request.files['image']
        np_image = np.array(Image.open(image_file).convert("RGB")).astype("uint8")

        ### Tree Detection ###
        MODEL_INFERENCE = {
            'patch_size': 400,
            'patch_overlap': 0.2,
            'iou_threshold': 0.05
        }

        # Predict the tree locations
        tree_predictions = tree_model.predict_tile(image=np_image, return_plot=True, **MODEL_INFERENCE)
        pretree = tree_model.predict_tile(image=np_image, return_plot=False, **MODEL_INFERENCE)

        # Count the number of detected trees
        detected_trees = pretree['boxes'] if isinstance(pretree, dict) else pretree
        tree_count = len(detected_trees)

        ### Waterbody Detection ###
        resized_image = cv2.resize(np_image, (128, 128)) / 255.0
        resized_image = np.expand_dims(resized_image, axis=0)
        predicted_mask = water_model.predict(resized_image)
        predicted_mask = np.squeeze(predicted_mask)
        predicted_mask_binary = (predicted_mask > 0.5).astype(np.uint8)
        original_size_mask = cv2.resize(predicted_mask_binary, (np_image.shape[1], np_image.shape[0]))

        # Create a color overlay for the waterbodies
        overlay = np.zeros_like(np_image)
        overlay[original_size_mask == 1] = [0, 0, 255]  # Red for water bodies
        color_overlay_image = cv2.addWeighted(np_image, 1, overlay, 0.3, 0)

        ### Convert Output to Base64 ###
        tree_detected_image = Image.fromarray(tree_predictions[:, :, ::-1])
        tree_img_io = io.BytesIO()
        tree_detected_image.save(tree_img_io, 'PNG')
        tree_img_io.seek(0)
        tree_base64_image = base64.b64encode(tree_img_io.getvalue()).decode('utf-8')

        overlay_img = Image.fromarray(cv2.cvtColor(color_overlay_image, cv2.COLOR_BGR2RGB))
        overlay_img_io = io.BytesIO()
        overlay_img.save(overlay_img_io, 'PNG')
        overlay_img_io.seek(0)
        water_overlay_base64_image = base64.b64encode(overlay_img_io.getvalue()).decode('utf-8')

        return jsonify({
            "message": "Image processed successfully",
            "tree_count": tree_count,
            "tree_detected_image": tree_base64_image,
            "water_overlay_image": water_overlay_base64_image
        }), 200

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
