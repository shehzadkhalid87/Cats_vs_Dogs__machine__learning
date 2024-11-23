import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Load your trained model
model = load_model('cat_vs_dog_model.keras')  # Or your model's file path

# Define the prediction function
def predict(image):
    img = image.resize((300, 300))  # Resize to the expected input size for the model
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)  # Get the model's prediction
    return "Dog" if prediction[0] > 0.5 else "Cat"  # Classify based on the output

# Create a Gradio interface
interface = gr.Interface(
    fn=predict,  # Function that will be called to process the image
    inputs="image",  # Input type is an image
    outputs="label",  # Output type is a label (Cat or Dog)
    title="Cat vs Dog Classifier",  # Interface title
    description="Upload an image to classify it as a cat or a dog."  # Description
)

# Launch the interface
interface.launch(share=True)
