from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
RANDOM_STATE = 42
IMAGE_SIZE = 224
CLASS_NAMES = ["negative", "positive"]

