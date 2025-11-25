import mlflow
import pandas as pd

# GANTI path model sesuai run kamu
MODEL_PATH = r"mlruns/171511009199790507/d61d15f38dcc458bb173fecd98d1f8be/artifacts/model"

# Load model
model = mlflow.pyfunc.load_model(MODEL_PATH)

# Dummy data (HARUS ADA kolom 'clean_title')
data = pd.DataFrame({
    "clean_title": [
        "government caught doing corruption scandal again",
        "artist wins award for best performance",
        "scientists discover cure for cancer in new study",
        "breaking city destroyed by alien invasion yesterday",
        "new iPhone announced with improved battery life",
        "president resigns after massive financial scandal",
        "research shows coffee improves memory and focus",
        "viral claim earth will end next week proven wrong",
        "local community opens free education program",
        "celebrities arrested for participating in secret cult"
    ]
})

# Inference
pred = model.predict(data)

print("Predictions:", pred)

# OPTIONAL: decode label angka ke teks
labels = {0: "FAKE", 1: "REAL"}
decoded = [labels[x] for x in pred]

print("Decoded:", decoded)
