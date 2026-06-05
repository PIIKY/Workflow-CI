"""Baseline model comparison for Dicoding MSML final submission.

This script intentionally uses manual MLflow logging so the reviewer can see
parameters, metrics, models, and additional artifacts for every candidate.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


RANDOM_STATE = 42
TARGET = "loan_status"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset_preprocessing"
ARTIFACT_DIR = BASE_DIR / "artifacts" / "baseline"


def configure_mlflow() -> None:
    """Configure MLflow for either DagsHub remote tracking or local fallback."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    dagshub_owner = os.getenv("DAGSHUB_REPO_OWNER")
    dagshub_repo = os.getenv("DAGSHUB_REPO_NAME")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "Credit Risk Prediction - Baseline")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif dagshub_owner and dagshub_repo:
        mlflow.set_tracking_uri(f"https://dagshub.com/{dagshub_owner}/{dagshub_repo}.mlflow")
    else:
        mlflow.set_tracking_uri(f"file:///{(BASE_DIR / 'mlruns').as_posix()}")

    active_tracking_uri = mlflow.get_tracking_uri()
    print(f"MLflow tracking URI: {active_tracking_uri}")
    print(f"MLflow experiment: {experiment_name}")
    if active_tracking_uri.startswith("file:"):
        print("WARNING: MLflow is using local file tracking. Set DagsHub env vars before running.")

    if os.getenv("MLFLOW_RUN_ID"):
        print("MLflow Project run detected; using the run created by `mlflow run`.")
        return

    mlflow.set_experiment(experiment_name)


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_path = DATA_DIR / "credit_risk_train.csv"
    test_path = DATA_DIR / "credit_risk_test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Processed dataset not found. Expected credit_risk_train.csv and "
            "credit_risk_test.csv inside Membangun_model/dataset_preprocessing."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    x_train = train_df.drop(columns=[TARGET]).astype(float)
    y_train = train_df[TARGET]
    x_test = test_df.drop(columns=[TARGET]).astype(float)
    y_test = test_df[TARGET]
    return x_train, x_test, y_train, y_test


def get_candidate_models(y_train: pd.Series) -> dict[str, Any]:
    negative, positive = np.bincount(y_train.astype(int))
    scale_pos_weight = negative / positive

    models: dict[str, Any] = {
        "LogisticRegression": LogisticRegression(
            C=1.0,
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            random_state=RANDOM_STATE,
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier

        models["LightGBM"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbose=-1,
        )
    except ImportError:
        pass

    return models


def predict_scores(model: Any, x_test: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]
    scores = model.decision_function(x_test)
    return (scores - scores.min()) / (scores.max() - scores.min())


def evaluate_model(model: Any, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = model.predict(x_test)
    y_score = predict_scores(model, x_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "average_precision": average_precision_score(y_test, y_score),
    }


def model_params(model: Any) -> dict[str, Any]:
    if hasattr(model, "get_params"):
        return model.get_params()
    return {}


def save_artifacts(
    model_name: str,
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    metrics: dict[str, float],
    artifact_root: Path = ARTIFACT_DIR,
) -> Path:
    artifact_dir = artifact_root / model_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(x_test)
    y_score = predict_scores(model, x_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    with open(artifact_dir / "classification_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    with open(artifact_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Good Loan", "Default"])
    display.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(artifact_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_score)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {metrics['roc_auc']:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(artifact_dir / "roc_curve.png", dpi=160)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, y_score)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"AP = {metrics['average_precision']:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(artifact_dir / "precision_recall_curve.png", dpi=160)
    plt.close()

    importance = None
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])

    if importance is not None:
        top_features = (
            pd.DataFrame({"feature": x_test.columns, "importance": importance})
            .sort_values("importance", ascending=False)
            .head(15)
            .sort_values("importance", ascending=True)
        )
        plt.figure(figsize=(8, 6))
        plt.barh(top_features["feature"], top_features["importance"], color="#2F6B7C")
        plt.xlabel("Importance")
        plt.title(f"Top Feature Importance - {model_name}")
        plt.tight_layout()
        plt.savefig(artifact_dir / "feature_importance.png", dpi=160)
        plt.close()
        top_features.sort_values("importance", ascending=False).to_csv(
            artifact_dir / "feature_importance.csv", index=False
        )

    return artifact_dir


def log_model_run(
    model_name: str,
    model: Any,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    with mlflow.start_run(run_name=f"baseline_{model_name}"):
        params = model_params(model)
        mlflow.log_params({key: value for key, value in params.items() if isinstance(value, (str, int, float, bool))})
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("target", TARGET)
        mlflow.log_param("train_rows", len(x_train))
        mlflow.log_param("test_rows", len(x_test))

        model.fit(x_train, y_train)
        metrics = evaluate_model(model, x_test, y_test)
        mlflow.log_metrics(metrics)

        signature = infer_signature(x_train.head(10), model.predict(x_train.head(10)))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=x_test.head(5),
        )

        artifact_dir = save_artifacts(model_name, model, x_test, y_test, metrics)
        mlflow.log_artifacts(str(artifact_dir), artifact_path="evaluation_artifacts")

        return {"model_name": model_name, **metrics}


def main() -> None:
    configure_mlflow()
    x_train, x_test, y_train, y_test = load_dataset()
    models = get_candidate_models(y_train)

    results = []
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        result = log_model_run(model_name, model, x_train, x_test, y_train, y_test)
        results.append(result)
        print(f"{model_name}: ROC-AUC={result['roc_auc']:.4f}, F1={result['f1_score']:.4f}")

    result_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    result_path = ARTIFACT_DIR / "model_comparison.csv"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(result_path, index=False)

    print("\nModel comparison sorted by ROC-AUC:")
    print(result_df.to_string(index=False))
    print(f"\nBest model: {result_df.iloc[0]['model_name']}")


if __name__ == "__main__":
    main()
