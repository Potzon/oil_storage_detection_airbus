"""Trocea imagenes 2560x2560 en tiles y exporta un dataset en formato YOLO."""
import shutil
from pathlib import Path

import cv2
import yaml

from src import dataset
from src.tiling import tile_origins

TILE_SIZE = 640
TILE_OVERLAP = 64
TILE_STRIDE = TILE_SIZE - TILE_OVERLAP

# Una caja se conserva en un tile si al menos esta fraccion de su area cae dentro
MIN_BOX_COVERAGE = 0.4

YOLO_DATASET_DIR_NAME = "yolo_dataset"


def tile_image_boxes(image_w, image_h, boxes, tile_size=TILE_SIZE, stride=TILE_STRIDE):
    """Genera (tile_x, tile_y, boxes_relativas_al_tile) para cada tile de la imagen."""
    xs = tile_origins(image_w, tile_size, stride)
    ys = tile_origins(image_h, tile_size, stride)

    for ty in ys:
        for tx in xs:
            tile_box = (tx, ty, tx + tile_size, ty + tile_size)
            tile_boxes = []
            for (x1, y1, x2, y2) in boxes:
                ix1, iy1 = max(x1, tile_box[0]), max(y1, tile_box[1])
                ix2, iy2 = min(x2, tile_box[2]), min(y2, tile_box[3])
                inter_w, inter_h = max(0, ix2 - ix1), max(0, iy2 - iy1)
                inter_area = inter_w * inter_h
                box_area = max(1e-6, (x2 - x1) * (y2 - y1))
                if inter_area / box_area < MIN_BOX_COVERAGE:
                    continue
                # recortar la caja al tile y expresarla en coords relativas al tile
                rx1, ry1 = max(x1, tile_box[0]) - tx, max(y1, tile_box[1]) - ty
                rx2, ry2 = min(x2, tile_box[2]) - tx, min(y2, tile_box[3]) - ty
                tile_boxes.append((rx1, ry1, rx2, ry2))
            yield tx, ty, tile_boxes


def _box_to_yolo_line(box, tile_size=TILE_SIZE, class_id=0):
    x1, y1, x2, y2 = box
    xc = (x1 + x2) / 2 / tile_size
    yc = (y1 + y2) / 2 / tile_size
    w = (x2 - x1) / tile_size
    h = (y2 - y1) / tile_size
    return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def build_yolo_dataset(train_ids, val_ids, boxes_by_image, output_dir=None, force=False):
    """Genera el dataset YOLO (tiles) para train y val, y devuelve la ruta al data.yaml."""
    output_dir = Path(output_dir) if output_dir else dataset.ROOT / YOLO_DATASET_DIR_NAME

    if output_dir.exists() and force:
        shutil.rmtree(output_dir)

    for split_name, image_ids in (("train", train_ids), ("val", val_ids)):
        images_out = output_dir / "images" / split_name
        labels_out = output_dir / "labels" / split_name
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        if any(images_out.iterdir()) and not force:
            continue

        for image_id in image_ids:
            image = dataset.read_image(dataset.image_path(image_id))
            h, w = image.shape[:2]
            boxes = boxes_by_image.get(image_id, [])

            for tx, ty, tile_boxes in tile_image_boxes(w, h, boxes):
                # en train nos quedamos solo con tiles que tengan al menos un tanque,
                # para no desbalancear el dataset con fondo vacio
                if split_name == "train" and not tile_boxes:
                    continue

                tile_img = image[ty:ty + TILE_SIZE, tx:tx + TILE_SIZE]
                tile_name = f"{image_id}_{tx}_{ty}"
                dataset.write_image(images_out / f"{tile_name}.jpg", tile_img)

                lines = [_box_to_yolo_line(b) for b in tile_boxes]
                (labels_out / f"{tile_name}.txt").write_text("\n".join(lines))

    data_yaml_path = output_dir / "data.yaml"
    data_yaml = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "oil-storage-tank"},
    }
    with open(data_yaml_path, "w") as f:
        yaml.safe_dump(data_yaml, f)

    return data_yaml_path
