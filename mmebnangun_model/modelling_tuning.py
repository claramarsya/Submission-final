import argparse
import os
from dotenv import load_dotenv

import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


def setup_dagshub():
    """
    Setup koneksi MLflow ke DagsHub menggunakan token saja.
    """
    load_dotenv()

    dagshub_user = os.getenv("DAGSHUB_USER")
    dagshub_repo = os.getenv("DAGSHUB_REPO")
    dagshub_token = os.getenv("DAGSHUB_TOKEN")

    if not dagshub_user or not dagshub_repo:
        raise Exception("[ERROR] DAGSHUB_USER dan DAGSHUB_REPO wajib diisi dalam .env")

    # Tracking URI MLflow
    tracking_uri = f"https://dagshub.com/{dagshub_user}/{dagshub_repo}.mlflow"
    mlflow.set_tracking_uri(tracking_uri)

    print(f"[DAGSHUB] Tracking MLflow ke: {tracking_uri}")

    # Jika token tersedia → pasang untuk autentikasi
    if dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        print("[DAGSHUB] Token autentikasi MLflow berhasil di-set.")
    else:
        print("[WARNING] DAGSHUB_TOKEN tidak ditemukan. Jika repo private, tracking bisa gagal.")

    # Init dagshub autopush
    dagshub.init(
        repo_name=dagshub_repo,
        repo_owner=dagshub_user,
        mlflow=True
    )

def train_model(data_path):
    print("=== MEMUAT DATASET PREPROCESSING ===")
    df = pd.read_csv(data_path)
    print(df.head())

    text_col = "clean_title"
    label_col = "real"

    print("\n=== CLEANING ===")

    if text_col not in df.columns:
        raise Exception(f"Kolom '{text_col}' tidak ditemukan dalam dataset!")

    df = df.dropna(subset=[text_col])
    df[text_col] = df[text_col].astype(str)
    df[text_col] = df[text_col].replace("", "empty")

    print("Sisa data:", len(df))

    print("\n=== SPLIT DATA ===")
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col],
        df[label_col],
        test_size=0.2,
        random_state=42,
        stratify=df[label_col]
    )
    print("Train:", len(X_train))
    print("Test :", len(X_test))

    # Setup DagsHub MLflow integration
    setup_dagshub()

    # Set experiment name
    mlflow.set_experiment("FakeNews_Clara")

    with mlflow.start_run():
        max_feat = 5000

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_feat)),
            ('clf', LogisticRegression(max_iter=200))
        ])

        # Log parameter experiment
        mlflow.log_param("tfidf_max_features", max_feat)
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        print("\n=== TRAINING MODEL ===")
        pipeline.fit(X_train, y_train)

        print("\n=== EVALUASI ===")
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print("Accuracy:", acc)
        print(classification_report(y_test, y_pred))

        mlflow.log_metric("accuracy", float(acc))

        # Log model ke DagsHub (via MLflow)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model"
        )

        print("\nModel berhasil disimpan ke DagsHub MLflow.")

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
