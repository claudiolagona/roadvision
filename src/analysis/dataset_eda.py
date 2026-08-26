from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OBJECTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "japan_objects_index.csv"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
)

TARGET_CLASSES = [
    "D00",
    "D10",
    "D20",
    "D40",
]


def load_target_objects():
    objects_df = pd.read_csv(
        OBJECTS_PATH
    )

    target_df = objects_df[
        objects_df["is_target"].eq(True)
        & objects_df["is_valid"].eq(True)
    ].copy()

    target_df[
        "relative_box_area"
    ] = (
        target_df["box_area"]
        / (
            target_df["image_width"]
            * target_df["image_height"]
        )
    )

    target_df[
        "aspect_ratio"
    ] = (
        target_df["box_width"]
        / target_df["box_height"]
    )

    return target_df


def print_statistics(
    target_df,
):
    print(
        "ROADVISION DATASET EDA\n"
    )

    print(
        f"Valid target objects: "
        f"{len(target_df)}"
    )

    print(
        "\nClass distribution:"
    )

    print(
        target_df["class_name"]
        .value_counts()
        .reindex(TARGET_CLASSES)
    )

    print(
        "\nMedian relative "
        "box area by class:"
    )

    print(
        target_df
        .groupby(
            "class_name"
        )["relative_box_area"]
        .median()
        .reindex(TARGET_CLASSES)
    )

    print(
        "\nMedian aspect "
        "ratio by class:"
    )

    print(
        target_df
        .groupby(
            "class_name"
        )["aspect_ratio"]
        .median()
        .reindex(TARGET_CLASSES)
    )


def save_class_distribution(
    target_df,
):
    counts = (
        target_df["class_name"]
        .value_counts()
        .reindex(TARGET_CLASSES)
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    counts.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title(
        "Class Distribution"
    )

    ax.set_xlabel(
        "Damage class"
    )

    ax.set_ylabel(
        "Number of objects"
    )

    ax.tick_params(
        axis="x",
        rotation=0,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "class_distribution.png",
        dpi=150,
    )

    plt.close(fig)


def save_relative_area_plot(
    target_df,
):
    data = [
        target_df.loc[
            (
                target_df[
                    "class_name"
                ]
                == class_name
            ),
            "relative_box_area",
        ]
        for class_name
        in TARGET_CLASSES
    ]

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.boxplot(
        data,
        tick_labels=TARGET_CLASSES,
        showfliers=False,
    )

    ax.set_title(
        "Relative Bounding-Box "
        "Area by Class"
    )

    ax.set_xlabel(
        "Damage class"
    )

    ax.set_ylabel(
        "Bounding-box area / "
        "image area"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "relative_box_area_by_class.png",
        dpi=150,
    )

    plt.close(fig)


def save_aspect_ratio_plot(
    target_df,
):
    data = [
        target_df.loc[
            (
                target_df[
                    "class_name"
                ]
                == class_name
            ),
            "aspect_ratio",
        ]
        for class_name
        in TARGET_CLASSES
    ]

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.boxplot(
        data,
        tick_labels=TARGET_CLASSES,
        showfliers=False,
    )

    ax.set_title(
        "Bounding-Box "
        "Aspect Ratio by Class"
    )

    ax.set_xlabel(
        "Damage class"
    )

    ax.set_ylabel(
        "Width / height"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "aspect_ratio_by_class.png",
        dpi=150,
    )

    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    target_df = (
        load_target_objects()
    )

    print_statistics(
        target_df
    )

    save_class_distribution(
        target_df
    )

    save_relative_area_plot(
        target_df
    )

    save_aspect_ratio_plot(
        target_df
    )

    print(
        f"\nFigures saved in: "
        f"{FIGURES_DIR}"
    )


if __name__ == "__main__":
    main()