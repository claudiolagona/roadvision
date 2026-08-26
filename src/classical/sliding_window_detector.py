from argparse import ArgumentParser
from pathlib import Path
import cv2
import joblib
import numpy as np
from src.classical.hog_features import (
    extract_hog_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "classical"
    / "hog_linear_svm.joblib"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "classical"
    / "hog_svm"
)

CLASS_NAMES = [
    "D00",
    "D10",
    "D20",
    "D40",
    "background",
]

BACKGROUND_ID = 4

WINDOW_SIZES = [
    (64, 64),
    (128, 64),
    (128, 128),
]

STEP_SIZE = 32

SCORE_THRESHOLD = 0.50

NMS_THRESHOLD = 0.30


def generate_windows(
    image,
    window_width,
    window_height,
):
    (
        image_height,
        image_width,
    ) = image.shape[:2]

    for y in range(
        0,
        (
            image_height
            - window_height
            + 1
        ),
        STEP_SIZE,
    ):
        for x in range(
            0,
            (
                image_width
                - window_width
                + 1
            ),
            STEP_SIZE,
        ):
            patch = image[
                y:y + window_height,
                x:x + window_width,
            ]

            yield (
                x,
                y,
                patch,
            )


def collect_detections(
    image,
    classifier,
):
    detections = []

    for (
        window_width,
        window_height,
    ) in WINDOW_SIZES:

        for (
            x,
            y,
            patch,
        ) in generate_windows(
            image,
            window_width,
            window_height,
        ):
            features = (
                extract_hog_features(
                    patch
                )
            )

            scores = (
                classifier
                .decision_function(
                    [features]
                )[0]
            )

            class_id = int(
                np.argmax(
                    scores
                )
            )

            score = float(
                scores[
                    class_id
                ]
            )

            if (
                class_id
                == BACKGROUND_ID
            ):
                continue

            if (
                score
                < SCORE_THRESHOLD
            ):
                continue

            detections.append(
                {
                    "class_id":
                        class_id,
                    "score":
                        score,
                    "box": [
                        x,
                        y,
                        window_width,
                        window_height,
                    ],
                }
            )

    return detections


def apply_nms(
    detections,
):
    final_detections = []

    for class_id in range(
        BACKGROUND_ID
    ):
        class_detections = [
            detection
            for detection
            in detections
            if (
                detection[
                    "class_id"
                ]
                == class_id
            )
        ]

        if not class_detections:
            continue

        boxes = [
            detection["box"]
            for detection
            in class_detections
        ]

        scores = [
            detection["score"]
            for detection
            in class_detections
        ]

        indexes = (
            cv2.dnn.NMSBoxes(
                boxes,
                scores,
                score_threshold=(
                    SCORE_THRESHOLD
                ),
                nms_threshold=(
                    NMS_THRESHOLD
                ),
            )
        )

        if len(indexes) == 0:
            continue

        for index in (
            np.asarray(
                indexes
            ).reshape(-1)
        ):
            final_detections.append(
                class_detections[
                    int(index)
                ]
            )

    return final_detections


def draw_detections(
    image,
    detections,
):
    output = image.copy()

    for detection in detections:
        (
            x,
            y,
            width,
            height,
        ) = detection["box"]

        class_name = (
            CLASS_NAMES[
                detection[
                    "class_id"
                ]
            ]
        )

        score = (
            detection["score"]
        )

        cv2.rectangle(
            output,
            (x, y),
            (
                x + width,
                y + height,
            ),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            output,
            (
                f"{class_name} "
                f"{score:.2f}"
            ),
            (
                x,
                max(
                    20,
                    y - 5,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return output


def detect(
    image,
    classifier,
):
    raw_detections = (
        collect_detections(
            image,
            classifier,
        )
    )

    final_detections = (
        apply_nms(
            raw_detections
        )
    )

    return final_detections


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help=(
            "Path to the road "
            "image to analyse."
        ),
    )

    args = parser.parse_args()

    image = cv2.imread(
        args.image
    )

    if image is None:
        raise FileNotFoundError(
            args.image
        )

    classifier = joblib.load(
        MODEL_PATH
    )

    detections = detect(
        image,
        classifier,
    )

    output = draw_detections(
        image,
        detections,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULTS_DIR
        / "sliding_window_example.jpg"
    )

    cv2.imwrite(
        str(output_path),
        output,
    )

    print(
        "HOG + SVM "
        "SLIDING-WINDOW DETECTOR\n"
    )

    print(
        "Detections after NMS: "
        f"{len(detections)}"
    )

    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()