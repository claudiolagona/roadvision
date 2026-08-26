from pathlib import Path
import shutil
import pandas as pd
from src.preprocessing.yolo_format import (
    CLASS_TO_ID,
    voc_box_to_yolo,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_IMAGES_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "RDD2022"
    / "Japan"
    / "train"
    / "images"
)

OBJECTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "japan_objects_index.csv"
)

SPLIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "japan_split.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "yolo"
    / "roadvision_japan"
)

DATA_YAML_PATH = (
    OUTPUT_ROOT
    / "data.yaml"
)

SPLIT_TO_FOLDER = {
    "train": "train",
    "validation": "val",
    "test": "test",
}


def prepare_directories():
    if OUTPUT_ROOT.exists():
        shutil.rmtree(
            OUTPUT_ROOT
        )

    for folder in (
        SPLIT_TO_FOLDER.values()
    ):
        (
            OUTPUT_ROOT
            / "images"
            / folder
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            OUTPUT_ROOT
            / "labels"
            / folder
        ).mkdir(
            parents=True,
            exist_ok=True,
        )


def make_label_lines(
    objects_df,
):
    lines = []

    for _, obj in (
        objects_df.iterrows()
    ):
        class_id = (
            CLASS_TO_ID[
                obj["class_name"]
            ]
        )

        (
            x_center,
            y_center,
            width,
            height,
        ) = voc_box_to_yolo(
            xmin=obj["xmin"],
            ymin=obj["ymin"],
            xmax=obj["xmax"],
            ymax=obj["ymax"],
            image_width=(
                obj["image_width"]
            ),
            image_height=(
                obj["image_height"]
            ),
        )

        lines.append(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

    return lines


def write_data_yaml():
    content = f'''path: "{OUTPUT_ROOT.as_posix()}"
train: images/train
val: images/val
test: images/test

names:
  0: D00
  1: D10
  2: D20
  3: D40
'''

    DATA_YAML_PATH.write_text(
        content,
        encoding="utf-8",
    )


def main():
    objects_df = pd.read_csv(
        OBJECTS_PATH
    )

    split_df = pd.read_csv(
        SPLIT_PATH
    )

    valid_targets = objects_df[
        objects_df["is_target"].eq(True)
        & objects_df["is_valid"].eq(True)
    ].copy()

    removed_targets = objects_df[
        objects_df["is_target"].eq(True)
        & ~objects_df["is_valid"].eq(True)
    ]

    objects_by_image = {
        image_name: group
        for image_name, group
        in valid_targets.groupby(
            "image"
        )
    }

    prepare_directories()

    statistics = {
        split_name: {
            "images": 0,
            "objects": 0,
        }
        for split_name
        in SPLIT_TO_FOLDER
    }

    for _, row in (
        split_df.iterrows()
    ):
        image_name = row["image"]
        split_name = row["split"]

        folder = (
            SPLIT_TO_FOLDER[
                split_name
            ]
        )

        source_image = (
            RAW_IMAGES_DIR
            / image_name
        )

        destination_image = (
            OUTPUT_ROOT
            / "images"
            / folder
            / image_name
        )

        if not source_image.exists():
            raise FileNotFoundError(
                f"Missing image: "
                f"{source_image}"
            )

        shutil.copy2(
            source_image,
            destination_image,
        )

        image_objects = (
            objects_by_image.get(
                image_name
            )
        )

        label_path = (
            OUTPUT_ROOT
            / "labels"
            / folder
            / (
                f"{Path(image_name).stem}"
                ".txt"
            )
        )

        if image_objects is None:
            label_lines = []

        else:
            label_lines = (
                make_label_lines(
                    image_objects
                )
            )

        label_path.write_text(
            "\n".join(
                label_lines
            ),
            encoding="utf-8",
        )

        statistics[
            split_name
        ]["images"] += 1

        statistics[
            split_name
        ]["objects"] += len(
            label_lines
        )

    write_data_yaml()

    print(
        "ROADVISION YOLO DATASET\n"
    )

    for (
        split_name,
        values,
    ) in statistics.items():

        print(
            f"{split_name}: "
            f"{values['images']} images, "
            f"{values['objects']} objects"
        )

    print(
        "\nInvalid target "
        "objects skipped: "
        f"{len(removed_targets)}"
    )

    print(
        f"Dataset saved in: "
        f"{OUTPUT_ROOT}"
    )

    print(
        f"YOLO config: "
        f"{DATA_YAML_PATH}"
    )


if __name__ == "__main__":
    main()