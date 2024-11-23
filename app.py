from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
model = load_model('cat_vs_dog_model.keras')  # Or .h5 if that’s your format

# Check if the 'uploads' directory exists, create it if it doesn't
uploads_dir = './uploads'
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

def predict_label(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return "Dog" if prediction[0] > 0.5 else "Cat"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        file_path = os.path.join(uploads_dir, file.filename)  # Save file in 'uploads' folder
        file.save(file_path)
        label = predict_label(file_path)
        return render_template("result.html", label=label)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
