import cv2
import numpy as np
from skimage.feature import hog


PATCH_SIZE = (64, 64)

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)


def extract_hog_features(image):

    if image is None:
        raise ValueError("Input image cannot be None.")

    resized = cv2.resize(
        image,
        PATCH_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    grayscale = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY,
    )

    features = hog(
        grayscale,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )

    return features.astype(
        np.float32
    )