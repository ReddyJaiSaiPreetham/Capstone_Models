from flask import Flask, render_template, request
import PyPDF2
import docx
import re
import joblib  # For loading the trained model
import os


app = Flask(__name__)

uploads = os.path.join(app.instance_path, 'uploads')
os.makedirs(uploads, exist_ok=True)


# Load the trained model (Ensure the model file is in the same directory)
model = joblib.load('kidney.pkl')

# List of features expected from the form
features = [
    'age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
    'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria',
    'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
    'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count',
    'red_blood_cell_count', 'hypertension', 'diabetes_mellitus',
    'coronary_artery_disease', 'appetite', 'peda_edema', 'aanemia'
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
        'blood_pressure': r'Blood Pressure:\s*(\d+)',
        'specific_gravity': r'Specific Gravity:\s*([\d.]+)',
        'albumin': r'Albumin:\s*(\d+)',
        'sugar': r'Sugar:\s*(\d+)',
        'red_blood_cells': r'Red Blood Cells:\s*(abnormal|normal)',
        'pus_cell': r'Pus Cell:\s*(abnormal|normal)',
        'pus_cell_clumps': r'Pus Cell Clumps:\s*(present|not present|absent)',
        'bacteria': r'Bacteria:\s*(present|not present|absent)',
        'blood_glucose_random': r'Blood Glucose Random:\s*(\d+)',
        'blood_urea': r'Blood Urea:\s*(\d+)',
        'serum_creatinine': r'Serum Creatinine:\s*([\d.]+)',
        'sodium': r'Sodium:\s*(\d+)',
        'potassium': r'Potassium:\s*([\d.]+)',
        'haemoglobin': r'Haemoglobin:\s*([\d.]+)',
        'packed_cell_volume': r'Packed Cell Volume:\s*(\d+)',
        'white_blood_cell_count': r'White Blood Cell Count:\s*(\d+)',
        'red_blood_cell_count': r'Red Blood Cell Count:\s*([\d.]+)',
        'hypertension': r'Hypertension:\s*(yes|no)',
        'diabetes_mellitus': r'Diabetes Mellitus:\s*(yes|no)',
        'coronary_artery_disease': r'Coronary Artery Disease:\s*(yes|no)',
        'appetite': r'Appetite:\s*(good|poor)',
        'peda_edema': r'Peda Edema:\s*(yes|no)',
        'aanemia': r'Aanemia:\s*(yes|no)'
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
        if value.lower() in ['abnormal', 'present', 'yes', 'poor']:
            processed_data.append(1)
        elif value.lower() in ['normal', 'not present', 'no', 'good', 'absent']:
            processed_data.append(0)
        else:
            try:
                processed_data.append(float(value))
            except (TypeError, ValueError):
                processed_data.append(0.0)

    # Model prediction
    prediction = model.predict([processed_data])[0]

    # Prediction result and precautions
    prediction_text = "Positive (Kidney Disease Detected)" if prediction == 0 else "Negative (No Kidney Disease)"
    precautions = [
        "Maintain a healthy diet low in sodium.",
        "Monitor blood pressure regularly.",
        "Stay hydrated and avoid smoking."
    ] if prediction == 0 else ["No immediate precautions needed. Continue regular health check-ups."]

    return render_template('home.html', features=features, extracted_data=input_data, prediction_text=prediction_text, precautions=precautions)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
