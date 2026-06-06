"""Hyperparameter tuning with manual MLflow logging.

Default tuning target is XGBoost because it produced the strongest ROC-AUC
in baseline comparison. Set TUNING_MODEL=GradientBoosting if you need a
pure scikit-learn fallback.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from modelling import (
    RANDOM_STATE,
    TARGET,
    BASE_DIR,
    configure_mlflow,
    evaluate_model,
    load_dataset,
    save_artifacts,
)


TUNING_ARTIFACT_DIR = BASE_DIR / "artifacts" / "tuning"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def build_search_space(model_name: str, y_train: pd.Series) -> tuple[Any, dict[str, list[Any]]]:
    if model_name == "GradientBoosting":
        estimator = GradientBoostingClassifier(random_state=RANDOM_STATE)
        param_distributions = {
            "n_estimators": [120, 180, 240, 300],
            "learning_rate": [0.03, 0.05, 0.08, 0.1],
            "max_depth": [2, 3, 4],
            "min_samples_leaf": [10, 20, 40],
            "subsample": [0.75, 0.9, 1.0],
            "max_features": ["sqrt", None],
        }
        return estimator, param_distributions

    if model_name == "XGBoost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "TUNING_MODEL=XGBoost requires xgboost. Install requirements.txt first."
            ) from exc

        negative, positive = np.bincount(y_train.astype(int))
        scale_pos_weight = negative / positive
        estimator = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        param_distributions = {
            "n_estimators": [200, 300, 400, 500],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.02, 0.05, 0.08, 0.1],
            "subsample": [0.75, 0.85, 1.0],
            "colsample_bytree": [0.75, 0.85, 1.0],
            "min_child_weight": [1, 3, 5],
            "reg_lambda": [1.0, 2.0, 5.0],
        }
        return estimator, param_distributions

    raise ValueError("Unsupported TUNING_MODEL. Use GradientBoosting or XGBoost.")


def save_tuning_metadata(
    search: RandomizedSearchCV,
    model_name: str,
    metrics: dict[str, float],
) -> Path:
    artifact_dir = TUNING_ARTIFACT_DIR / f"tuned_{model_name}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    cv_results.to_csv(artifact_dir / "cv_results.csv", index=False)

    metadata = {
        "model_name": model_name,
        "search_method": "RandomizedSearchCV",
        "scoring": "roc_auc",
        "cv_folds": 3,
        "n_iter": search.n_iter,
        "best_cv_roc_auc": to_jsonable(search.best_score_),
        "best_params": to_jsonable(search.best_params_),
        "test_metrics": to_jsonable(metrics),
    }
    with open(artifact_dir / "tuning_summary.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return artifact_dir


def main() -> None:
    configure_mlflow()
    x_train, x_test, y_train, y_test = load_dataset()

    model_name = os.getenv("TUNING_MODEL", "XGBoost")
    n_iter = int(os.getenv("TUNING_N_ITER", "24"))
    estimator, param_distributions = build_search_space(model_name, y_train)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        verbose=1,
        random_state=RANDOM_STATE,
        return_train_score=True,
    )

    mlflow.autolog()
    with mlflow.start_run(run_name=f"tuning_{model_name}", nested=True):
        search.fit(x_train, y_train)
        best_model = search.best_estimator_
        metrics = evaluate_model(best_model, x_test, y_test)

        eval_artifacts = save_artifacts(
            f"tuned_{model_name}",
            best_model,
            x_test,
            y_test,
            metrics,
            artifact_root=TUNING_ARTIFACT_DIR,
        )
        metadata_artifacts = save_tuning_metadata(search, model_name, metrics)
        mlflow.log_artifacts(str(eval_artifacts), artifact_path="evaluation_artifacts")
        mlflow.log_artifacts(str(metadata_artifacts), artifact_path="tuning_artifacts")

    print(f"Best model: {model_name}")
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    print("Best params:")
    print(json.dumps(to_jsonable(search.best_params_), indent=2))
    print("Test metrics:")
    print(json.dumps(to_jsonable(metrics), indent=2))


if __name__ == "__main__":
    main()
