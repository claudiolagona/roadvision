from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "classical"
)

TRAIN_PATH = (
    DATA_DIR
    / "hog_train_features.npz"
)

VALIDATION_PATH = (
    DATA_DIR
    / "hog_validation_features.npz"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "classical"
)

MODEL_PATH = (
    MODEL_DIR
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

RANDOM_STATE = 42


def load_dataset(path):
    data = np.load(path)

    return (
        data["X"],
        data["y"],
    )


def save_confusion_matrix(
    y_true,
    y_pred,
):
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=range(
            len(CLASS_NAMES)
        ),
        normalize="true",
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    image = ax.imshow(
        matrix,
        vmin=0,
        vmax=1,
    )

    ax.set_xticks(
        range(
            len(CLASS_NAMES)
        )
    )

    ax.set_yticks(
        range(
            len(CLASS_NAMES)
        )
    )

    ax.set_xticklabels(
        CLASS_NAMES
    )

    ax.set_yticklabels(
        CLASS_NAMES
    )

    ax.set_xlabel(
        "Predicted class"
    )

    ax.set_ylabel(
        "True class"
    )

    ax.set_title(
        "HOG + Linear SVM - "
        "Normalized Confusion Matrix"
    )

    for row in range(
        matrix.shape[0]
    ):
        for column in range(
            matrix.shape[1]
        ):
            ax.text(
                column,
                row,
                (
                    f"{matrix[row, column]:.2f}"
                ),
                ha="center",
                va="center",
            )

    fig.colorbar(
        image,
        ax=ax,
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR
        / "confusion_matrix_normalized.png",
        dpi=150,
    )

    plt.close(fig)


def main():
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_train, y_train = (
        load_dataset(
            TRAIN_PATH
        )
    )

    (
        X_validation,
        y_validation,
    ) = load_dataset(
        VALIDATION_PATH
    )

    print(
        "HOG + LINEAR SVM\n"
    )

    print(
        f"Train: "
        f"{X_train.shape}"
    )

    print(
        f"Validation: "
        f"{X_validation.shape}"
    )

    classifier = LinearSVC(
        C=1.0,
        max_iter=5000,
        random_state=RANDOM_STATE,
    )

    classifier.fit(
        X_train,
        y_train,
    )

    predictions = (
        classifier.predict(
            X_validation
        )
    )

    accuracy = accuracy_score(
        y_validation,
        predictions,
    )

    print(
        f"\nAccuracy: "
        f"{accuracy:.4f}"
    )

    print(
        "\nClassification report:\n"
    )

    print(
        classification_report(
            y_validation,
            predictions,
            labels=range(
                len(CLASS_NAMES)
            ),
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    joblib.dump(
        classifier,
        MODEL_PATH,
    )

    save_confusion_matrix(
        y_validation,
        predictions,
    )

    print(
        f"Saved model: "
        f"{MODEL_PATH}"
    )

    print(
        f"Saved results: "
        f"{RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()