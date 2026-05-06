from __future__ import annotations

from bc_predictor.config import ARTIFACT_DIR
from bc_predictor.metrics import save_metrics
from bc_predictor.tabular_model import save_tabular_model, train_wisconsin_baseline


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model, metrics = train_wisconsin_baseline()
    save_tabular_model(model, ARTIFACT_DIR / "wisconsin_baseline.joblib")
    save_metrics(metrics, ARTIFACT_DIR / "wisconsin_metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()

