from flask import Flask, request, jsonify
import mlflow
import pandas as pd

MODEL_PATH = r"mlruns/171511009199790507/d61d15f38dcc458bb173fecd98d1f8be/artifacts/model"

try:
    model = mlflow.pyfunc.load_model(MODEL_PATH)
    print(f"[INFO] Model loaded from: {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    model = None

LABELS = {0: "FAKE", 1: "REAL"}

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json(force=True)

    if not data:
        return jsonify({"error": "Request must be JSON"}), 400

    if "text" in data:
        texts = [data["text"]]
        single = True
    elif "texts" in data:
        texts = data["texts"]
        single = False
    else:
        return jsonify({"error": "JSON must contain 'text' or 'texts'"}), 400

    df = pd.DataFrame({"clean_title": texts})

    try:
        pred = model.predict(df)
    except Exception as e:
        return jsonify({"error": f"Prediction error: {e}"}), 500

    decoded = [LABELS[x] for x in pred]

    response = {
        "predictions": pred.tolist(),
        "decoded_labels": decoded
    }

    if single:
        response["predictions"] = response["predictions"][0]
        response["decoded_labels"] = response["decoded_labels"][0]

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
