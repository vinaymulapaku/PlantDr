import h5py
import gradio as gr
from tensorflow.keras.utils import img_to_array, load_img
from keras.models import load_model
import numpy as np
from deep_translator import GoogleTranslator

# Load the pre-trained model from the local path
model_path = 'sugar.h5'

# Check if the model is loading correctly
try:
    with h5py.File(model_path, 'r+') as f:
        if 'groups' in f.attrs['model_config']:  
            model_config_string = f.attrs['model_config'] 
            model_config_string = model_config_string.replace('"groups": 1,', '')
            model_config_string = model_config_string.replace('"groups": 1}', '}')
            f.attrs['model_config'] = model_config_string.encode('utf-8')

    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

def predict_disease(image_file, model, all_labels, target_language):
    try:
        # Load and preprocess the image
        print(f"Received image file: {image_file}")
        img = load_img(image_file, target_size=(224, 224))  # Ensure image size matches model input
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize the image

        # Predict the class
        predictions = model.predict(img_array)
        confidence_scores = predictions[0]
        max_confidence = np.max(confidence_scores)
        confidence_threshold = 0.98  # Require at least 98% confidence

        # Check if the confidence is too low
        if max_confidence < confidence_threshold:
            print(f"Prediction confidence ({max_confidence:.2f}) is too low.")
            return f"""
                <h3 style="color:red; text-align:center;">
                    The uploaded image does not appear to be a sugarcane plant or has low clarity. 
                    Please upload a clearer image of a sugarcane plant.
                </h3>
            """

        # Get the predicted class index
        predicted_class = np.argmax(confidence_scores)

        # Get the predicted class label
        predicted_label = all_labels[predicted_class]

        # Check for irrelevant or non-sugarcane-related predictions
        if predicted_label not in all_labels:
            print("The image does not match sugarcane diseases.")
            return f"""
                <h3 style="color:red; text-align:center;">
                    The uploaded image does not match any sugarcane diseases. 
                    Please ensure you upload a sugarcane plant image.
                </h3>
            """

        # Translate the predicted label
        translated_label = GoogleTranslator(source='en', target=target_language).translate(predicted_label)

        # Provide pesticide information based on the predicted label
        # (Use your existing logic here for each disease case)

        # Example for a matched class:
        if predicted_label == 'Sugarcane Yellow':
            pesticide_info = """
                <h2><center><b>Sugarcane Yellow</b></center></h2>
                <h4>PESTICIDES TO BE USED:</h4><br>
                <ul style="font-size:17px;margin-left:40px;">
                    <li>1. Insecticidal Soap</li>
                    <li>2. Pyrethroids</li>
                    <li>3. Imidacloprid</li>
                    <li>4. Bacillus thuringiensis</li>
                    <li>5. Spinosad</li>
                </ul><br>
                <center><p class="note" style="font-size:15px;"><b>* * * IMPORTANT NOTE * * *</b></p></center><br>
                <center><p style="font-size:13px;">Be sure to follow local regulations and guidelines for application</p></center>
            """
        elif predicted_label == 'Sugarcane Rust':
            pesticide_info = """
                <h2><center><b>Sugarcane Rust</b></center></h2>
                <h4>PESTICIDES TO BE USED:</h4><br>
                
                <ul style="font-size:17px;margin-left:40px;">
                    <li>1. Triadimefon</li>
                    <li>2. Chlorothalonil</li>
                    <li>3. Tebuconazole</li>
                    <li>4. Propiconazole</li>
                </ul><br>
                <center><p class="note" style="font-size:15px;"><b>* * * IMPORTANT NOTE * * *</b></p></center><br>
                <center><p style="font-size:13px;">Be sure to follow local regulations and guidelines for application</p></center>
            """
        elif predicted_label == 'Sugarcane RedRot':
            pesticide_info = """
                <h2><center><b>Sugarcane RedRot</b></center></h2>
                <h4>PESTICIDES TO BE USED:</h4><br>
                
                <ul style="font-size:17px;margin-left:40px;">
                    <li>1. Mancozeb</li>
                    <li>2. Chlorothalonil</li>
                    <li>3. Tebuconazole</li>
                    <li>4. Carbendazim</li>
                </ul><br>
                <center><p class="note" style="font-size:15px;"><b>* * * IMPORTANT NOTE * * *</b></p></center><br>
                <center><p style="font-size:13px;">Be sure to follow local regulations and guidelines for application</p></center>
            """
        elif predicted_label == 'Sugarcane Mosaic':
            pesticide_info = """
                <h2><center><b>Sugarcane Mosaic</b></center></h2>
                <h4>PESTICIDES TO BE USED:</h4><br>
                
                <ul style="font-size:17px;margin-left:40px;">
                    <li>1. Horticultural Oil</li>
                    <li>2. Spinosad</li>
                    <li>3. Pyrethrin </li>
                    <li>4. Neem Oil</li>
                    <li>5. Imidacloprid</li>
                </ul><br>
                <center><p class="note" style="font-size:15px;"><b>* * * IMPORTANT NOTE * * *</b></p></center><br>
                <center><p style="font-size:13px;">Be sure to follow local regulations and guidelines for application</p></center>
            """
            
        elif predicted_label == 'Sugarcane Healthy':
            pesticide_info = """<h2><center><b>Sugarcane Healthy</b></center></h2>
            <h5> No pesticides needed"""
        else:
            pesticide_info = 'No pesticide information available.'

        # Translate pesticide info to the selected language
        translated_pesticide_info = GoogleTranslator(source='en', target=target_language).translate(pesticide_info)

        # Return translated label and pesticide information
        return f"{translated_pesticide_info}"

    except Exception as e:
        print(f"Error during prediction: {e}")
        return f"<h3>Error: {e}</h3>"


# List of class labels
all_labels = [
    'Sugarcane Yellow',
    'Sugarcane Rust',
    'Sugarcane RedRot','Sugarcane Mosaic',
    'Sugarcane Healthy'
]

# Language codes and their full names (display full names in dropdown)
language_choices = {
    'hi': 'Hindi',
    'te': 'Telugu',
    'en': 'English',
    'ml': 'Malayalam',
    'ta': 'Tamil',
    'bn': 'Bengali',
    'gu': 'Gujarati',
    'kn': 'Kannada',
    'mr': 'Marathi'
}

# Mapping full names back to their corresponding language code
full_to_code = {value: key for key, value in language_choices.items()}

# Create a dropdown of full language names, using the full name in the UI
languages = list(language_choices.values())  # List of full language names

# Define the Gradio interface
def gradio_predict(image_file, target_language):
    # Map full name back to language code for translation
    language_code = full_to_code.get(target_language, 'en')
    return predict_disease(image_file, model, all_labels, language_code)

# Create the Gradio interface
gr_interface = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Image(type="filepath"),  # Image input for disease prediction
        gr.Dropdown(label="Select language", choices=languages, value='English')  # Language selection dropdown with full names
    ],
    outputs="html",  # Output will be in HTML (translated text)
    title="Sugarcane Disease Predictor",
    description="Upload an image of a plant to predict the disease and get the translated label and pesticide information in the selected language."
)

# Launch the Gradio app
gr_interface.launch()