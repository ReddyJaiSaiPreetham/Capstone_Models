import os
import pickle
import numpy as np
import re
from flask import Flask, request, render_template
from PyPDF2 import PdfReader
from docx import Document

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load pre-trained model and scaler
svmodel = pickle.load(open('svmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

# Keywords for matching
KEYWORDS = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer",
    "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
]

# Home page route
@app.route('/')
def home():
    return render_template('home.html', keywords=KEYWORDS)

# File upload and extraction route
@app.route('/extract', methods=['POST'])
def extract_data():
    extracted_data = {}

    # File Upload Handling
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        safe_filename = file.filename.encode('utf-8', errors='ignore').decode()
        filename = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        file.save(filename)

        # Detect file type and extract text
        if file.filename.endswith('.pdf'):
            text = extract_text_from_pdf(filename)
        elif file.filename.endswith('.docx'):
            text = extract_text_from_docx(filename)
        else:
            return render_template('home.html', error='Unsupported file format')

        # Extract data using regular expressions
        extracted_data = regex_extract(text)

    return render_template(
        "home.html",
        extracted_data=extracted_data,
        keywords=KEYWORDS
    )

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    input_data = []
    extracted_data = {}

    # Collect data from form fields (manual or auto-filled)
    for keyword in KEYWORDS:
        value = request.form.get(keyword)
        try:
            if value and value.strip() != '':
                input_data.append(float(value))
                extracted_data[keyword] = float(value)
            else:
                input_data.append(0.0)
        except (ValueError, TypeError):
            input_data.append(0.0)

    # Scale and predict
    input_scaled = scaler.transform(np.array(input_data).reshape(1, -1))
    prediction = svmodel.predict(input_scaled)[0]

    # Prediction result
    result = "Positive for Parkinson’s Disease" if prediction == 1 else "Negative for Parkinson’s Disease"

    # Precautions for positive result
    precautions = []
    if prediction == 1:
        precautions = [
            "🧘 Engage in regular physical exercise and therapy.",
            "🥗 Follow a balanced diet rich in fiber and antioxidants.",
            "💊 Take medications as prescribed by your doctor.",
            "🩺 Schedule regular neurological checkups.",
            "🧠 Practice mental exercises and stress management."
        ]

    return render_template(
        "home.html",
        prediction_text=f"Prediction: {result}",
        precautions=precautions if prediction == 1 else None,
        extracted_data=extracted_data,
        keywords=KEYWORDS
    )

# Extract text from PDF
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        try:
            page_text = page.extract_text()
            if page_text:
                cleaned_text = re.sub(r"[•\t\r\u200B]", " ", page_text).strip()
                text += cleaned_text + "\n"
        except UnicodeEncodeError:
            continue
    return text.strip()

# Extract text from DOCX
def extract_text_from_docx(file):
    doc = Document(file)
    text = ""
    for para in doc.paragraphs:
        try:
            cleaned_text = re.sub(r"[•\t\r\u200B]", " ", para.text).strip()
            text += cleaned_text + "\n"
        except UnicodeEncodeError:
            continue
    return text.strip()

# Regex-based data extraction
def regex_extract(text):
    extracted_data = {}

    # Clean hidden characters, normalize spaces, and handle special formatting
    text = re.sub(r"[•\t\r\u200B\u00A0]", " ", text)  # Remove unwanted characters
    text = re.sub(r"\s+", " ", text)  # Normalize spaces
    text = text.replace("–", "-")  # Replace special minus signs with standard minus

    # Regex pattern for accurate matching of negative, decimal, or scientific values
    pattern = re.compile(
        r"(MDVP:Fo\(Hz\)|MDVP:Fhi\(Hz\)|MDVP:Flo\(Hz\)|MDVP:Jitter\(%\)|"
        r"MDVP:Jitter\(Abs\)|MDVP:RAP|MDVP:PPQ|Jitter:DDP|MDVP:Shimmer|MDVP:Shimmer\(dB\)|"
        r"Shimmer:APQ3|Shimmer:APQ5|MDVP:APQ|Shimmer:DDA|NHR|HNR|RPDE|DFA|Spread1|Spread2|D2|PPE)"
        r"[:\s]*"  # Matches colon or spaces
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",  # Captures negative, decimal, or scientific notation
        flags=re.IGNORECASE
    )

    matches = pattern.findall(text)

    # Log extracted values for debugging
    print("🔍 Extracted Matches:", matches)

    # Store extracted values in the dictionary
    for match in matches:
        keyword, value = match
        try:
            extracted_data[keyword] = float(value)
        except ValueError:
            extracted_data[keyword] = None

    # Ensure all keywords are present
    for keyword in KEYWORDS:
        if keyword not in extracted_data:
            extracted_data[keyword] = None

    # Log extracted data for debugging
    print("✅ Final Extracted Data:", extracted_data)

    return extracted_data


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
