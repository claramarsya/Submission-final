import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

def train_model(data_path):
    print("=== MEMUAT DATASET PREPROCESSING ===")
    df = pd.read_csv(data_path)
    print(df.head())
    
    text_col = "clean_title"
    label_col = "real"
    
    # === SAFETY CLEANING (FIX ERROR) ===
    print("\n=== FIXING NaN / BAD STRINGS ===")
    
    # pastikan kolom ada
    if text_col not in df.columns:
        raise Exception(f"Kolom '{text_col}' tidak ditemukan! Apakah file hasil preprocessing benar?")
    
    # drop baris dengan missing text
    df = df.dropna(subset=[text_col])
    
    # convert ke string
    df[text_col] = df[text_col].astype(str)
    
    # ganti string kosong
    df[text_col] = df[text_col].replace("", "empty")
    
    print("Sisa data setelah cleaning:", len(df))


    print("\n=== SPLIT DATA ===")
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col],
        df[label_col],
        test_size=0.2,
        random_state=42,
        stratify=df[label_col]
    )
    print("Train size:", len(X_train))
    print("Test size :", len(X_test))

    # MLflow Setup
    mlflow.set_experiment("FakeNews_Clara")

    with mlflow.start_run():
        # Model pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', LogisticRegression(max_iter=200))
        ])

        print("\n=== TRAINING MODEL ===")
        pipeline.fit(X_train, y_train)

        print("\n=== EVALUASI MODEL ===")
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print("Accuracy:", acc)
        print(classification_report(y_test, y_pred))

        # Log metrics ke MLflow
        mlflow.log_metric("accuracy", acc)

        # Log model
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        print("\nModel berhasil disimpan ke MLflow.")

    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="FakeNewsNet_preprocessing.csv",
        help="Path ke dataset hasil preprocessing"
    )
    args = parser.parse_args()

    train_model(args.data_path)