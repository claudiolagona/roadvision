from pathlib import Path
import xml.etree.ElementTree as ET

def parse_annotation(xml_path: Path) -> dict:

    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text

    size = root.find("size")

    width = int(size.find("width").text)
    height = int(size.find("height").text)

    depth_element = size.find("depth")
    depth = (
        int(depth_element.text)
        if depth_element is not None
        else None
    )

    objects = []

    for obj in root.findall("object"):
        class_name = obj.find("name").text

        truncated_element = obj.find("truncated")
        difficult_element = obj.find("difficult")

        truncated = (
            int(truncated_element.text)
            if truncated_element is not None
            else 0
        )
        difficult = (
            int(difficult_element.text)
            if difficult_element is not None
            else 0
        )

        bbox = obj.find("bndbox")

        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        objects.append(
            {
                "class_name": class_name,
                "bbox": {
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                },
                "truncated": truncated,
                "difficult": difficult
            }
        )

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "depth": depth,
        "objects": objects
    }