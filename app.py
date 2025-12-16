import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get form data
    TV = float(request.form["TV"])
    Radio = float(request.form["Radio"])
    Newspaper = int(request.form["Newspaper"])
   

    # Prepare features for prediction
   
    feature_names = ["TV", "Radio", "Newspaper"]
    features = pd.DataFrame([[TV, Radio, Newspaper]], columns=feature_names)

    # Predict charges
    prediction = model.predict(features)
    #  Format to float and 2 decimal places
    formatted_prediction = f"The sales generated is ${round(float(prediction), 2)}"


    return render_template("result.html", prediction=formatted_prediction)


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')
