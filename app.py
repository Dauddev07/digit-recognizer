import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageOps
import tensorflow as tf

app = Flask(__name__)

# Load the pre-trained model on startup
MODEL_PATH = "model/digit_recognizer.keras"

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")
print("Server is ready to make predictions.")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()
        image_data = data["image"].split(",")[1]
        image_bytes = base64.b64decode(image_data)

        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("L")

        image_array_check = np.array(image)
        mean_pixel = image_array_check.mean()
        if mean_pixel > 127:
            image = ImageOps.invert(image)

        img_array = np.array(image)
        rows = np.any(img_array > 30, axis=1)
        cols = np.any(img_array > 30, axis=0)

        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            image = image.crop((cmin, rmin, cmax, rmax))
            w, h = image.size
            pad = int(max(w, h) * 0.25)
            padded = Image.new("L", (w + pad * 2, h + pad * 2), 0)
            padded.paste(image, (pad, pad))
            image = padded

        image = image.resize((28, 28), Image.LANCZOS)
        image_array = np.array(image).astype("float32") / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)

        predictions = model.predict(image_array, verbose=0)
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_digit]) * 100
        all_probabilities = {
            str(i): round(float(predictions[0][i]) * 100, 2)
            for i in range(10)
        }

        return jsonify({
            "digit": predicted_digit,
            "confidence": round(confidence, 2),
            "all_probabilities": all_probabilities
        })

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Starting Flask server...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)