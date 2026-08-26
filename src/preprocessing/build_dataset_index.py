from pathlib import Path
import pandas as pd
from src.preprocessing.pascal_voc import parse_annotation


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "RDD2022"
    / "Japan"
    / "train"
)

IMAGES_DIR = DATASET_ROOT / "images"
ANNOTATIONS_DIR = DATASET_ROOT / "annotations" / "xmls"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OBJECTS_OUTPUT_PATH = OUTPUT_DIR / "japan_objects_index.csv"
IMAGES_OUTPUT_PATH = OUTPUT_DIR / "japan_images_index.csv"

TARGET_CLASSES = {"D00", "D10", "D20", "D40"}


def build_indexes():
    object_rows = []
    image_rows = []

    xml_files = sorted(ANNOTATIONS_DIR.glob("*.xml"))

    for xml_path in xml_files:
        annotation = parse_annotation(xml_path)

        image_name = annotation["filename"]
        image_path = IMAGES_DIR / image_name

        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")

        valid_target_count = 0

        for obj in annotation["objects"]:
            class_name = obj["class_name"]
            bbox = obj["bbox"]

            xmin = bbox["xmin"]
            ymin = bbox["ymin"]
            xmax = bbox["xmax"]
            ymax = bbox["ymax"]

            box_width = xmax - xmin
            box_height = ymax - ymin
            box_area = box_width * box_height

            is_target = class_name in TARGET_CLASSES

            valid_geometry = (
                box_width > 0
                and box_height > 0
            )

            in_bounds = (
                xmin >= 0
                and ymin >= 0
                and xmax <= annotation["width"]
                and ymax <= annotation["height"]
            )

            is_valid = valid_geometry and in_bounds

            if is_target and is_valid:
                valid_target_count += 1

            object_rows.append(
                {
                    "image": image_name,
                    "xml": xml_path.name,
                    "image_width": annotation["width"],
                    "image_height": annotation["height"],
                    "class_name": class_name,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "box_width": box_width,
                    "box_height": box_height,
                    "box_area": box_area,
                    "is_target": is_target,
                    "valid_geometry": valid_geometry,
                    "in_bounds": in_bounds,
                    "is_valid": is_valid,
                }
            )

        image_rows.append(
            {
                "image": image_name,
                "xml": xml_path.name,
                "width": annotation["width"],
                "height": annotation["height"],
                "valid_target_object_count": valid_target_count,
            }
        )

    return pd.DataFrame(object_rows), pd.DataFrame(image_rows)


def main():
    objects_df, images_df = build_indexes()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    objects_df.to_csv(
        OBJECTS_OUTPUT_PATH,
        index=False,
    )

    images_df.to_csv(
        IMAGES_OUTPUT_PATH,
        index=False,
    )

    valid_targets = objects_df[
        objects_df["is_target"].eq(True)
        & objects_df["is_valid"].eq(True)
    ]

    invalid_targets = objects_df[
        objects_df["is_target"].eq(True)
        & ~objects_df["is_valid"].eq(True)
    ]

    print("ROADVISION DATASET INDEX\n")
    print(f"Images: {len(images_df)}")
    print(f"Objects: {len(objects_df)}")
    print(
        f"Valid target objects: "
        f"{len(valid_targets)}"
    )
    print(
        f"Invalid target objects: "
        f"{len(invalid_targets)}"
    )

    print(f"\nSaved: {OBJECTS_OUTPUT_PATH}")
    print(f"Saved: {IMAGES_OUTPUT_PATH}")


if __name__ == "__main__":
    main()