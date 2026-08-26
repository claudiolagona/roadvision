from pathlib import Path
from ultralytics import YOLO
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "yolo"
    / "best.pt"
)

DATA_CONFIG = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "yolo"
    / "roadvision_japan"
    / "data.yaml"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "yolo"
)

IMAGE_SIZE = 640
BATCH_SIZE = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"


def main():
    model = YOLO(
        str(MODEL_PATH)
    )

    model.val(
        data=str(
            DATA_CONFIG
        ),

        split="test",

        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        device=DEVICE,

        plots=True,

        project=str(
            RESULTS_DIR
        ),

        name=(
            "final_test_yolo11s"
        ),

        exist_ok=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()