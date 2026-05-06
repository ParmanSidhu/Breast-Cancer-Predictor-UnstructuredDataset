from __future__ import annotations

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bc_predictor.config import RANDOM_STATE
from bc_predictor.metrics import binary_classification_metrics


def train_wisconsin_baseline() -> tuple[Pipeline, dict]:
    dataset = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=dataset.target,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = binary_classification_metrics(
        np.asarray(y_test),
        predictions,
        probabilities,
    )
    metrics["dataset"] = "Wisconsin Breast Cancer structured baseline"
    metrics["note"] = (
        "This baseline uses engineered tabular features and is not directly "
        "comparable to raw image classification."
    )
    return model, metrics


def save_tabular_model(model: Pipeline, path) -> None:
    joblib.dump(model, path)

