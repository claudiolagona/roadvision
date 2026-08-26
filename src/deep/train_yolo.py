from argparse import ArgumentParser
from pathlib import Path
from ultralytics import YOLO
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

MODEL_NAME = "yolo11s.pt"

IMAGE_SIZE = 640
BATCH_SIZE = 8
EPOCHS = 50
PATIENCE = 10
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 42


def train_baseline():
    model = YOLO(
        MODEL_NAME
    )

    model.train(
        data=str(DATA_CONFIG),

        epochs=EPOCHS,
        patience=PATIENCE,

        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        device=DEVICE,
        seed=SEED,

        pretrained=True,

        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,

        degrees=0.0,
        translate=0.0,
        scale=0.0,

        shear=0.0,
        perspective=0.0,

        flipud=0.0,
        fliplr=0.0,

        mosaic=0.0,
        mixup=0.0,

        project=str(
            RESULTS_DIR
        ),

        name=(
            "baseline_yolo11s"
        ),

        plots=True,
        verbose=True,
    )


def train_augmented():
    model = YOLO(
        MODEL_NAME
    )

    model.train(
        data=str(DATA_CONFIG),

        epochs=EPOCHS,
        patience=PATIENCE,

        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        device=DEVICE,
        seed=SEED,

        pretrained=True,

        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,

        translate=0.05,
        scale=0.10,

        fliplr=0.5,

        flipud=0.0,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,

        mosaic=0.0,
        mixup=0.0,

        project=str(
            RESULTS_DIR
        ),

        name=(
            "augmented_yolo11s"
        ),

        plots=True,
        verbose=True,
    )


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--experiment",
        choices=[
            "baseline",
            "augmented",
        ],
        required=True,
    )

    args = parser.parse_args()

    if (
        args.experiment
        == "baseline"
    ):
        train_baseline()

    else:
        train_augmented()


if __name__ == "__main__":
    main()