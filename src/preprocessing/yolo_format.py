CLASS_TO_ID = {
    "D00": 0,
    "D10": 1,
    "D20": 2,
    "D40": 3,
}


def voc_box_to_yolo(
    xmin,
    ymin,
    xmax,
    ymax,
    image_width,
    image_height,
):
    box_width = xmax - xmin
    box_height = ymax - ymin

    if (
        box_width <= 0
        or box_height <= 0
    ):
        raise ValueError(
            "Bounding box must "
            "have positive size."
        )

    x_center = (
        (xmin + xmax)
        / 2
        / image_width
    )

    y_center = (
        (ymin + ymax)
        / 2
        / image_height
    )

    width = (
        box_width
        / image_width
    )

    height = (
        box_height
        / image_height
    )

    return (
        x_center,
        y_center,
        width,
        height,
    )