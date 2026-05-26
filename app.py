from flask import Flask, request, jsonify
import joblib
import numpy as np

model = joblib.load("iris_model.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return "Iris Classifier API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        
        data = request.get_json(force=True)
        
        features = np.array(data["features"]).reshape(1, -1)
        
        prediction = model.predict(features)[0]
        
        classes = ["setosa", "versicolor", "virginica"]
        result = {"prediction": classes[prediction]}
        
        return jsonify(result)
        
    except Exception as e:
       
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)

    
