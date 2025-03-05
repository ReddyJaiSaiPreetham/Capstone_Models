from flask import Flask, render_template, request
import PyPDF2
import docx
import re
import joblib  # For loading the trained model
import os
import numpy as np

app = Flask(__name__)

uploads = os.path.join(app.instance_path, 'uploads')
os.makedirs(uploads, exist_ok=True)


# Load the trained model (Ensure the model file is in the same directory)
import joblib

model = joblib.load('liver_disease.pkl')
print(type(model))




# List of features expected from the form
features = [
    'age', 'gender', 'total_bilirubin', 'direct_bilirubin', 'alkaline_phosphotase',
    'alamine_aminotransferase', 'aspartate_aminotransferase', 'total_proteins',
    'albumin', 'albumin_and_globulin_ratio'
]

# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text() + ' '
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + ' '
    return text

# Function to extract data from text using regex
def extract_data_from_text(text):
    extracted_data = {}
    patterns = {
        'age': r'Age:\s*(\d+)',
        'gender': r'Gender:\s*(Male|Female)',
        'total_bilirubin': r'Total Bilirubin:\s*([\d.]+)',
        'direct_bilirubin': r'Direct Bilirubin:\s*([\d.]+)',
        'alkaline_phosphotase': r'Alkaline Phosphatase:\s*(\d+)',
        'alamine_aminotransferase': r'ALT:\s*(\d+)',
        'aspartate_aminotransferase': r'AST:\s*(\d+)',
        'total_proteins': r'Total Proteins:\s*([\d.]+)',
        'albumin': r'Albumin:\s*([\d.]+)',
        'albumin_and_globulin_ratio': r'Albumin & Globulin Ratio:\s*([\d.]+)'
    }

    for feature, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        extracted_data[feature] = match.group(1) if match else None

    return extracted_data

# Home Route
@app.route('/', methods=['GET'])
def home():
    return render_template('home.html', features=features, extracted_data={}, prediction_text=None, precautions=None)

# File Upload Route
@app.route('/extract', methods=['POST'])
def extract():
    file = request.files['file']
    if file.filename.endswith('.pdf'):
        text = extract_text_from_pdf(file)
    elif file.filename.endswith('.docx'):
        text = extract_text_from_docx(file)
    else:
        return "Unsupported file format. Please upload a PDF or DOCX."

    extracted_data = extract_data_from_text(text)
    return render_template('home.html', features=features, extracted_data=extracted_data, prediction_text=None, precautions=None)

# Prediction Route (Using Trained Model)
@app.route('/predict', methods=['POST'])
def predict():
    input_data = {feature: request.form.get(feature) for feature in features}

    # Convert input data for the model
    processed_data = []
    for feature in features:
        value = input_data.get(feature)
        if value and value.lower() in ['male']:
            processed_data.append(1)  # Encode Male as 1
        elif value and value.lower() in ['female']:
            processed_data.append(0)  # Encode Female as 0
        else:
            try:
                processed_data.append(float(value))  # Convert numbers to float
            except (TypeError, ValueError):
                processed_data.append(0.0)  # Default to 0.0 if invalid input

    # Model prediction
    input_array = np.array(processed_data).reshape(1, -1)  # Corrected input formatting
    prediction = model.predict(input_array)[0]  

    # Interpret Prediction
    prediction_text = "Positive (Liver Disease Detected)" if prediction == 1 else "Negative (No Liver Disease)"
    precautions = [
        "Limit alcohol intake and eat a balanced diet.",
        "Exercise regularly and maintain a healthy weight.",
        "Monitor liver enzyme levels periodically."
    ] if prediction == 1 else ["No immediate precautions needed. Continue regular health check-ups."]

    return render_template('home.html', features=features, extracted_data=input_data, prediction_text=prediction_text, precautions=precautions)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
