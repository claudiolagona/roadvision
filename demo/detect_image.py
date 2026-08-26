from argparse import ArgumentParser
from pathlib import Path
import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "yolo"
    / "best.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "demo"
)

IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.25


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return YOLO(str(MODEL_PATH))


def detect_image(
    model,
    image,
):
    results = model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
    )

    return results[0]


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the road image.",
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    model = load_model()

    result = detect_image(
        model,
        image,
    )

    annotated_image = (
        result.plot()
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_detected.jpg"
    )

    cv2.imwrite(
        str(output_path),
        annotated_image,
    )

    print("ROADVISION - YOLO DETECTION\n")

    print(
        f"Detections: "
        f"{len(result.boxes)}"
    )

    print(
        f"Saved: {output_path}"
    )

    cv2.imshow(
        "RoadVision",
        annotated_image,
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()