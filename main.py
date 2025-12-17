from flask import Flask, render_template, request, jsonify
import pickle
import os

app = Flask(__name__)

model = None   # ← important

def load_model():
    global model
    if model is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "model", "model.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    load_model()   # ← load only when needed

    data = request.get_json()
    study_hours = float(data["study_hours"])
    attendance = float(data["attendance"])
    previous_score = float(data["previous_score"])

    prediction = model.predict([[study_hours, attendance, previous_score]])

    return jsonify({
        "predicted_final_score": round(float(prediction[0]), 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
