from argparse import ArgumentParser
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from src.classical.hog_features import (
    PATCH_SIZE,
    extract_hog_features,
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "classical"
)

CLASS_TO_ID = {
    "D00": 0,
    "D10": 1,
    "D20": 2,
    "D40": 3,
    "background": 4,
}

BACKGROUND_ID = (
    CLASS_TO_ID["background"]
)

RANDOM_SEED = 42


def load_split_data(
    split_name,
):
    objects_df = pd.read_csv(
        OBJECTS_PATH
    )

    split_df = pd.read_csv(
        SPLIT_PATH
    )

    split_images = split_df[
        split_df["split"]
        == split_name
    ].copy()

    targets_df = objects_df.merge(
        split_images[["image"]],
        on="image",
        how="inner",
    )

    targets_df = targets_df[
        targets_df["is_target"].eq(True)
        & targets_df["is_valid"].eq(True)
    ].copy()

    negative_images = (
        split_images[
            split_images[
                "combination"
            ]
            == "NONE"
        ]["image"]
        .tolist()
    )

    return (
        targets_df,
        negative_images,
    )


def extract_positive_samples(
    targets_df,
):
    features = []
    labels = []

    for (
        image_name,
        group,
    ) in targets_df.groupby(
        "image"
    ):
        image_path = (
            RAW_IMAGES_DIR
            / image_name
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise FileNotFoundError(
                image_path
            )

        for _, obj in (
            group.iterrows()
        ):
            xmin = int(
                obj["xmin"]
            )

            ymin = int(
                obj["ymin"]
            )

            xmax = int(
                obj["xmax"]
            )

            ymax = int(
                obj["ymax"]
            )

            crop = image[
                ymin:ymax,
                xmin:xmax,
            ]

            if crop.size == 0:
                continue

            features.append(
                extract_hog_features(
                    crop
                )
            )

            labels.append(
                CLASS_TO_ID[
                    obj["class_name"]
                ]
            )

    return (
        features,
        labels,
    )


def random_background_crop(
    image,
    rng,
):
    (
        patch_width,
        patch_height,
    ) = PATCH_SIZE

    (
        image_height,
        image_width,
    ) = image.shape[:2]

    if (
        image_width < patch_width
        or image_height < patch_height
    ):
        raise ValueError(
            "Image is smaller "
            "than the HOG patch size."
        )

    max_x = (
        image_width
        - patch_width
    )

    max_y = (
        image_height
        - patch_height
    )

    x = int(
        rng.integers(
            0,
            max_x + 1,
        )
    )

    y = int(
        rng.integers(
            0,
            max_y + 1,
        )
    )

    return image[
        y:y + patch_height,
        x:x + patch_width,
    ]


def extract_background_samples(
    image_names,
    sample_count,
    rng,
):
    if not image_names:
        raise ValueError(
            "No negative images "
            "available for this split."
        )

    features = []
    labels = []

    shuffled_images = list(
        rng.permutation(
            image_names
        )
    )

    base_samples = (
        sample_count
        // len(shuffled_images)
    )

    extra_samples = (
        sample_count
        % len(shuffled_images)
    )

    for (
        index,
        image_name,
    ) in enumerate(
        shuffled_images
    ):
        samples_from_image = (
            base_samples
        )

        if index < extra_samples:
            samples_from_image += 1

        if samples_from_image == 0:
            continue

        image_path = (
            RAW_IMAGES_DIR
            / image_name
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise FileNotFoundError(
                image_path
            )

        for _ in range(
            samples_from_image
        ):
            crop = (
                random_background_crop(
                    image,
                    rng,
                )
            )

            features.append(
                extract_hog_features(
                    crop
                )
            )

            labels.append(
                BACKGROUND_ID
            )

    return (
        features,
        labels,
    )


def build_dataset(
    split_name,
):
    rng = (
        np.random.default_rng(
            RANDOM_SEED
        )
    )

    (
        targets_df,
        negative_images,
    ) = load_split_data(
        split_name
    )

    (
        positive_features,
        positive_labels,
    ) = extract_positive_samples(
        targets_df
    )

    (
        background_features,
        background_labels,
    ) = (
        extract_background_samples(
            negative_images,
            sample_count=len(
                positive_features
            ),
            rng=rng,
        )
    )

    X = np.asarray(
        (
            positive_features
            + background_features
        ),
        dtype=np.float32,
    )

    y = np.asarray(
        (
            positive_labels
            + background_labels
        ),
        dtype=np.int64,
    )

    order = rng.permutation(
        len(y)
    )

    X = X[order]
    y = y[order]

    return X, y


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "validation",
        ],
        required=True,
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X, y = build_dataset(
        args.split
    )

    output_path = (
        OUTPUT_DIR
        / (
            f"hog_{args.split}"
            "_features.npz"
        )
    )

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
    )

    (
        unique_ids,
        counts,
    ) = np.unique(
        y,
        return_counts=True,
    )

    print(
        f"HOG DATASET - "
        f"{args.split.upper()}\n"
    )

    print(
        f"X shape: {X.shape}"
    )

    print(
        f"y shape: {y.shape}"
    )

    print(
        "\nClass distribution:"
    )

    id_to_class = {
        class_id: class_name
        for class_name, class_id
        in CLASS_TO_ID.items()
    }

    for (
        class_id,
        count,
    ) in zip(
        unique_ids,
        counts,
    ):
        print(
            f"{id_to_class[int(class_id)]}: "
            f"{count}"
        )

    print(
        f"\nSaved: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()