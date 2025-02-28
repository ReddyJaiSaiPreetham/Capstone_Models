from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model
with open("lung_cancer_prediction_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Column order for model input
columns = [
    "GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY",
    "PEER_PRESSURE", "CHRONIC DISEASE", "FATIGUE", "ALLERGY",
    "WHEEZING", "ALCOHOL CONSUMING", "COUGHING", "SHORTNESS OF BREATH",
    "SWALLOWING DIFFICULTY", "CHEST PAIN"
]

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Get inputs and convert to model-compatible format
            user_inputs = [int(request.form[col]) for col in columns]
            user_inputs = np.array(user_inputs).reshape(1, -1)

            # Make prediction
            prediction = model.predict(user_inputs)[0]
            result = "Positive for Lung Cancer" if prediction == 1 else "Negative for Lung Cancer"

            return render_template("home.html", prediction=result)
        except Exception as e:
            return render_template("home.html", error=str(e))

    return render_template("home.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
