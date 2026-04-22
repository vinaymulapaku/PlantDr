import h5py
import gradio as gr
from tensorflow.keras.utils import img_to_array, load_img
from keras.models import load_model
import numpy as np
from deep_translator import GoogleTranslator

# Load the pre-trained model from the local path
model_path = 'Mango.h5'

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
        predictions = model.predict(img_array)
        confidence_threshold = 0.7  # Require at least 98% confidence
        confidence_scores = predictions[0]
        max_confidence = np.max(confidence_scores)

        if max_confidence < confidence_threshold:
            print(f"Prediction confidence ({max_confidence:.2f}) is too low.")
            return f"""
                <h3 style="color:red; text-align:center;"> 
                Please upload a clearer image of the plant.
                </h3>
                """
        predicted_class = np.argmax(predictions[0])
        
        # Get the predicted class label
        predicted_label = all_labels[predicted_class]
        
        # Translate the predicted label to the selected language
        translated_label = GoogleTranslator(source='en', target=target_language).translate(predicted_label)
        
        # Provide pesticide information based on the predicted label
        if predicted_label == 'Mango Anthracrose':
            pesticide_info = """
                <h2><center><b>Mango Anthracrose</b></center></h2>
                <h4>PESTICIDES TO BE USED:</h4><br>
                <ul style="font-size:17px;margin-left:40px;">
                    <li>1. Mancozeb</li>
                    <li>2. Azoxystrobin</li>
                    <li>3. Carbendazim</li>
                    <li>4. Propiconazole</li>
                    <li>5. Thiophanate-methyl</li>
                    <li>6. Copper Sulfate</li>
                </ul><br>
                <center><p class="note" style="font-size:15px;"><b>* * * IMPORTANT NOTE * * *</b></p></center><br>
                <center><p style="font-size:13px;">Be sure to follow local regulations and guidelines for application</p></center>
            """
        elif predicted_label == 'Mango Bacterial Canker':
            pesticide_info = """<h2><center><b>Mango Bacterial Canker</b></center></h2>
                <h4>PESTICIDES TO BE USED:</h4><br>
                <ul style="font-size:17px;margin-left:40px;">
                    <li>1. Copper Hydroxide</li>
                    <li>2. Copper Oxychloride</li>
                    <li>3. Streptomycin</li>
                    <li>4. oxytetracycline</li>
                    <li>5. Neem oil</li>
                    <li>6. Garlic oil</li>
                </ul><br>
                <center><p class="note" style="font-size:15px;"><b>* * * IMPORTANT NOTE * * *</b></p></center><br>
                <center><p style="font-size:13px;">Be sure to follow local regulations and guidelines for application</p></center>
            """
        elif predicted_label == 'Mango Cutting Weevil':
            pesticide_info = """<h2><center><b>Mango Cutting Weevil</b></center></h2>
                <h4>PESTICIDES TO BE USED:</h4><br>
                <ul style="font-size:17px;margin-left:40px;">
                    <li>1. Imidacloprid</li>
                    <li>2. Thiamethoxam</li>
                    <li>3. Chlorpyrifos</li>
                    <li>4. Lambda-cyhalothrin</li>
                    <li>5. Fipronil</li>
                    <li>6. Neem oil</li>
                </ul><br>
                <center><p class="note" style="font-size:15px;"><b>* * * IMPORTANT NOTE * * *</b></p></center><br>
                <center><p style="font-size:13px;">Be sure to follow local regulations and guidelines for application</p></center>


            """
        elif predicted_label == 'Mango Die Back':
            pesticide_info = """<h2><center><b>Mango Die Back</b></center></h2>
                <h4>PESTICIDES TO BE USED:</h4><br>
                <ul style="font-size:17px;margin-left:40px;">
                    <li>1. Carbendazim</li>
                    <li>2. Mancozeb</li>
                    <li>3. Azoxystrobin</li>
                    <li>4. Triazole</li>
                    <li>5. Potassium bicarbonate</li>
                    <li>6. Sodium bicarbonate</li>
                </ul>
                <br>
                <center><p class="note" style="font-size:15px;"><b>* * * IMPORTANT NOTE * * *</b></p></center><br>
                <center><p style="font-size:13px;">Be sure to follow local regulations and guidelines for application</p></center>
            """
        elif predicted_label == 'Mango Gall Midge':
            pesticide_info = """<h2><center><b>Mango Gall Midge</b></center></h2>
                <h4>PESTICIDES TO BE USED:</h4><br>
                <ul style="font-size:17px;margin-left:40px;">
                    <li>1. Imidacloprid</li>
                    <li>2. Thiamethoxam</li>
                    <li>3. Chlorpyrifos</li>
                    <li>4. Lambda-cyhalothrin</li>
                    <li>5. Spinosad</li>
                    <li>6. Pyrethrin</li>
                </ul>
                <br>
                <center><p class="note" style="font-size:15px;"><b>* * * IMPORTANT NOTE * * *</b></p></center><br>
                <center><p style="font-size:13px;">Be sure to follow local regulations and guidelines for application</p></center>


            """
            
        elif predicted_label == 'Mango Healthy':
            pesticide_info = """<h2><center><b>Mango Healthy</b></center></h2>
            <h5> No pesticides needed"""
        elif predicted_label == 'Mango Powdery Mildew':
            pesticide_info = """<h2><center><b>Mango Powdery Mildew</b></center></h2>
                <h4>PESTICIDES TO BE USED:</h4><br>
                <ul style="font-size:17px;margin-left:40px;">
                    <li>1. Sulfur</li>
                    <li>2. Bicarbonates</li>
                    <li>3. Myclobutanil</li>
                    <li>4. Triadimefon</li>
                    <li>5. Propiconazole</li>
                    <li>6. Azoxystrobin</li>
                </ul>
                <br>
                <center><p class="note" style="font-size:15px;"><b>* * * IMPORTANT NOTE * * *</b></p></center><br>
                <center><p style="font-size:13px;">Be sure to follow local regulations and guidelines for application</p></center>
            """
        elif predicted_label == 'Mango Sooty Mould':
            pesticide_info = """<h2><center><b>Mango Sooty Mould</b></center></h2>
                <h4>PESTICIDES TO BE USED:</h4><br>
                <ul style="font-size:17px;margin-left:40px;">
                    <li>1. Imidacloprid (Neonicotinoid)</li>
                    <li>2. Thiamethoxam (Neonicotinoid)</li>
                    <li>3. Bifenthrin (Pyrethroid)</li>
                    <li>4. Lambda-cyhalothrin (Pyrethroid)</li>
                    <li>5. Insecticidal soap</li>
                    <li>6. Horticultural oil</li>
                </ul>
                <br>
                <center><p class="note" style="font-size:15px;"><b>* * * IMPORTANT NOTE * * *</b></p></center><br>
                <center><p style="font-size:13px;">Be sure to follow local regulations and guidelines for application</p></center>"""
        else:
            pesticide_info = 'No pesticide information available.'

        print(f"Pesticide Info (Before Translation): {pesticide_info}")

        # Translate the pesticide information to the selected language
        translated_pesticide_info = GoogleTranslator(source='en', target=target_language).translate(pesticide_info)
        print(f"Translated Pesticide Info: {translated_pesticide_info}")

        # Return translated label and pesticide information with associated styling
        predicted_label_html = f"""
        
        {translated_pesticide_info}
        """
        return predicted_label_html

    except Exception as e:
        print(f"Error during prediction: {e}")
        return f"<h3>Error: {e}</h3>"

# List of class labels
all_labels = [
    'Mango Anthracrose',
    'Mango Bacterial Canker',
    'Mango Cutting Weevil',
    'Mango Die Back',
    'Mango Gall Midge',
    'Mango Healthy',
    'Mango Powdery Mildew',
    'Mango Sooty Mould'
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
    title="Plant Disease Predictor",
    description="Upload an image of a plant to predict the disease and get the translated label and pesticide information in the selected language."
)

# Launch the Gradio app
gr_interface.launch()