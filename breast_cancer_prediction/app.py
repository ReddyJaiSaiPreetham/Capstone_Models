from flask import Flask, render_template, request
import PyPDF2
import docx
import re
import joblib  # For loading the trained model
import os

app = Flask(__name__)

uploads = os.path.join(app.instance_path, 'uploads')
os.makedirs(uploads, exist_ok=True)

# Load the trained breast cancer model
model = joblib.load('breast_cancer_ensemble.pkl')

# Features required for prediction
features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']  # Example


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
    'radius_mean': r'Radius Mean:\s*([\d.]+)',
    'texture_mean': r'Texture Mean:\s*([\d.]+)',
    'perimeter_mean': r'Perimeter Mean:\s*([\d.]+)',
    'area_mean': r'Area Mean:\s*([\d.]+)',
    'smoothness_mean': r'Smoothness Mean:\s*([\d.]+)',
    'compactness_mean': r'Compactness Mean:\s*([\d.]+)',
    'concavity_mean': r'Concavity Mean:\s*([\d.]+)',
    'concave points_mean': r'Concave Points Mean:\s*([\d.]+)',
    'symmetry_mean': r'Symmetry Mean:\s*([\d.]+)',
    'fractal_dimension_mean': r'Fractal Dimension Mean:\s*([\d.]+)',
    'radius_se': r'Radius SE:\s*([\d.]+)',
    'texture_se': r'Texture SE:\s*([\d.]+)',
    'perimeter_se': r'Perimeter SE:\s*([\d.]+)',
    'area_se': r'Area SE:\s*([\d.]+)',
    'smoothness_se': r'Smoothness SE:\s*([\d.]+)',
    'compactness_se': r'Compactness SE:\s*([\d.]+)',
    'concavity_se': r'Concavity SE:\s*([\d.]+)',
    'concave points_se': r'Concave Points SE:\s*([\d.]+)',
    'symmetry_se': r'Symmetry SE:\s*([\d.]+)',
    'fractal_dimension_se': r'Fractal Dimension SE:\s*([\d.]+)',
    'radius_worst': r'Radius Worst:\s*([\d.]+)',
    'texture_worst': r'Texture Worst:\s*([\d.]+)',
    'perimeter_worst': r'Perimeter Worst:\s*([\d.]+)',
    'area_worst': r'Area Worst:\s*([\d.]+)',
    'smoothness_worst': r'Smoothness Worst:\s*([\d.]+)',
    'compactness_worst': r'Compactness Worst:\s*([\d.]+)',
    'concavity_worst': r'Concavity Worst:\s*([\d.]+)',
    'concave points_worst': r'Concave Points Worst:\s*([\d.]+)',
    'symmetry_worst': r'Symmetry Worst:\s*([\d.]+)',
    'fractal_dimension_worst': r'Fractal Dimension Worst:\s*([\d.]+)'
}


    for feature, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        extracted_data[feature] = float(match.group(1)) if match else None

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

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    input_data = {feature: request.form.get(feature) for feature in features}

    # Convert input data for the model
    processed_data = []
    for feature in features:
        value = input_data.get(feature)
        try:
            processed_data.append(float(value))
        except (TypeError, ValueError):
            processed_data.append(0.0)

    # Model prediction
    prediction = model.predict([processed_data])[0]

    # Prediction result and precautions
    if prediction == 0:
        prediction_text = "Benign (No Cancer Detected)"
        precautions = ["Continue regular screenings and maintain a healthy lifestyle."]
    else:
        prediction_text = "Malignant (Cancer Detected)"
        precautions = [
            "Consult an oncologist for further tests.",
            "Follow recommended treatments and screenings.",
            "Maintain a healthy diet and exercise regularly."
        ]

    return render_template('home.html', features=features, extracted_data=input_data, prediction_text=prediction_text, precautions=precautions)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
