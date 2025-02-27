from flask import Flask, render_template, request, redirect, url_for
import joblib
import os
import re
import PyPDF2
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = joblib.load('diabetes_model.sav')

# Features for diabetes prediction
features = [
    'Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin',
    'BMI', 'Diabetes Pedigree Function', 'Age'
]

def extract_text_from_pdf(filepath):
    text = ""
    with open(filepath, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
            except UnicodeDecodeError:
                continue
    return text
def extract_data(text):
    extracted_data = {feature: None for feature in features}

    # Updated regex patterns with flexible spacing and colon handling
    patterns = {
        'Pregnancies': r'Pregnancies[:\s]*([\d\.]+)',
        'Glucose': r'Glucose[:\s]*([\d\.]+)',
        'Blood Pressure': r'Blood[\s]*Pressure[:\s]*([\d\.]+)',
        'Skin Thickness': r'Skin[\s]*Thickness[:\s]*([\d\.]+)',
        'Insulin': r'Insulin[:\s]*([\d\.]+)',
        'BMI': r'BMI[:\s]*([\d\.]+)',
        'Diabetes Pedigree Function': r'Diabetes[\s]*Pedigree[\s]*Function[:\s]*([\d\.]+)',
        'Age': r'Age[:\s]*([\d\.]+)'
    }

    # Extract values using regex
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1)
            extracted_data[key] = float(value) if '.' in value else int(value)
        else:
            extracted_data[key] = 0  # Default to 0 if value isn't found

    print("✅ Extracted Data:", extracted_data)
    return extracted_data


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html', features=features, extracted_data=None)

@app.route('/extract', methods=['POST'])
def extract():
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)

        if file.filename.endswith('.pdf'):
            text = extract_text_from_pdf(filepath)
        else:
            return "❌ Unsupported file format. Please upload a PDF."
        
        extracted_data = extract_data(text)
        return render_template('home.html', features=features, extracted_data=extracted_data)
    return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
def predict():
    input_data = []
    for feature in features:
        value = request.form.get(feature)
        input_data.append(float(value) if value else 0)

    # Predict using the loaded model
    prediction = model.predict([input_data])[0]
    prediction_text = '⚠️ High Risk of Diabetes' if prediction == 1 else '✅ Low Risk of Diabetes'

    # Suggest precautions if high risk
    precautions = [
        '🟢 Follow a balanced diet low in sugar.',
        '🏃 Exercise regularly.',
        '💉 Monitor blood sugar levels frequently.',
        '🩺 Regular check-ups with your doctor.'
    ] if prediction == 1 else []

    return render_template('home.html', features=features, extracted_data=None, prediction_text=prediction_text, precautions=precautions)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
