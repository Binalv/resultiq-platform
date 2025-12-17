from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load trained model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- UI ----------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# ---------------- Health API ----------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "project": "ResultIQ Platform",
        "status": "Running"
    })

# ---------------- Prediction API ----------------
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        study_hours = data["study_hours"]
        attendance = data["attendance"]
        previous_score = data["previous_score"]

        prediction = model.predict([[study_hours, attendance, previous_score]])

        return jsonify({
            "predicted_final_score": round(float(prediction[0]), 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ---------------- Model Info API ----------------
@app.route("/api/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "algorithm": "Linear Regression",
        "features": ["study_hours", "attendance", "previous_score"],
        "target": "final_score"
    })

if __name__ == "__main__":
    app.run(debug=True)
