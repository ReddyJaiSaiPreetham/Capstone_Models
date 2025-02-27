from flask import Flask, render_template, request, redirect, url_for
import joblib
import os
import re
import PyPDF2
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = joblib.load('models/heart.pkl')

# Descriptive features
features = [
    'Age', 'Sex', 'Chest Pain Type', 'Resting Blood Pressure', 'Serum Cholesterol',
    'Fasting Blood Sugar', 'Resting ECG', 'Max Heart Rate Achieved', 'Exercise Induced Angina',
    'ST Depression', 'Slope of Peak Exercise', 'Number of Major Vessels Colored', 'Thalassemia'
]

# Categorical mappings
mappings = {
    'Sex': {'Male': 0, 'Female': 1},
    'Fasting Blood Sugar': {'Normal': 0, 'High': 1},
    'Exercise Induced Angina': {'No': 0, 'Yes': 1}
}

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

    # Patterns to extract numerical and categorical data
    patterns = {
        'Age': r'Age:\s*(\d+)',
        'Sex': r'Sex:\s*(Male|Female)',
        'Chest Pain Type': r'Chest Pain Type:\s*(\d+)',
        'Resting Blood Pressure': r'Resting Blood Pressure:\s*(\d+)',
        'Serum Cholesterol': r'Serum Cholesterol:\s*(\d+)',
        'Fasting Blood Sugar': r'Fasting Blood Sugar:\s*(Normal|High)',
        'Resting ECG': r'Resting ECG:\s*(\d+)',
        'Max Heart Rate Achieved': r'Max Heart Rate Achieved:\s*(\d+)',
        'Exercise Induced Angina': r'Exercise Induced Angina:\s*(Yes|No)',
        'ST Depression': r'ST Depression:\s*([\d\.]+)',
        'Slope of Peak Exercise': r'Slope of Peak Exercise:\s*(\d+)',
        'Number of Major Vessels Colored': r'Number of Major Vessels Colored:\s*(\d+)',
        'Thalassemia': r'Thalassemia:\s*(\d+)'  
    }

    # Extract values using regex patterns
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1)
            # Convert categorical values to numeric
            if key in mappings:
                extracted_data[key] = mappings[key].get(value, None)
            else:
                extracted_data[key] = float(value)
    
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
        if feature in ['Sex', 'Fasting Blood Sugar', 'Exercise Induced Angina']:
            value = 1 if value and value.lower() in ['female', 'high', 'yes'] else 0
        input_data.append(float(value) if value else 0)
    
    # Predict using the loaded model
    prediction = model.predict([input_data])[0]
    prediction_text = '⚠️ High Risk of Heart Disease' if prediction == 1 else '✅ Low Risk of Heart Disease'

    # Suggest precautions if high risk
    precautions = [
        '🟢 Maintain a healthy diet.',
        '🏃 Exercise regularly.',
        '🧂 Monitor cholesterol levels.',
        '🩺 Regular heart check-ups.'
    ] if prediction == 1 else []

    return render_template('home.html', features=features, extracted_data=None, prediction_text=prediction_text, precautions=precautions)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
