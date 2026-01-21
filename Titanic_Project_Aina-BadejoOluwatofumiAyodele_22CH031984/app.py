from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load saved model and encoders
model = pickle.load(open("model/titanic_survival_model.pkl", "rb"))
le_sex = pickle.load(open("model/le_sex.pkl", "rb"))
le_embarked = pickle.load(open("model/le_embarked.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

FEATURE_NAMES = ["Pclass", "Sex", "Age", "SibSp", "Embarked"]
NUMERIC_COLS = ["Age", "SibSp"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get form data
    pclass = int(request.form["Pclass"])
    sex_text = request.form["Sex"]
    age = float(request.form["Age"])
    sibsp = int(request.form["SibSp"])
    embarked_text = request.form["Embarked"]

    # Encode categorical safely
    sex = le_sex.transform([sex_text])[0]
    embarked = le_embarked.transform([embarked_text])[0]

    # Create DataFrame with correct feature names
    input_data = pd.DataFrame([[pclass, sex, age, sibsp, embarked]], columns=FEATURE_NAMES)
    
    # Scale numeric features
    input_data[NUMERIC_COLS] = scaler.transform(input_data[NUMERIC_COLS])

    # Predict
    prediction = model.predict(input_data)[0]
    result = "Survived" if prediction == 1 else "Did Not Survive"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
