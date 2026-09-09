"""Carga de annotations.csv y split train/val reproducible a nivel de imagen."""
import ast
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "airbus_oil_storage_detection"
IMAGES_DIR = DATA_DIR / "images"
EXTRAS_DIR = DATA_DIR / "extras"
ANNOTATIONS_CSV = DATA_DIR / "annotations.csv"
SPLIT_PATH = ROOT / "results" / "split.json"

VAL_FRACTION = 0.2
SPLIT_SEED = 42


def _normalize_box(bounds_str):
    x1, y1, x2, y2 = ast.literal_eval(bounds_str)
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def load_annotations():
    """Devuelve dict image_id -> lista de cajas (x1, y1, x2, y2), normalizadas."""
    df = pd.read_csv(ANNOTATIONS_CSV)
    df["box"] = df["bounds"].apply(_normalize_box)
    boxes_by_image = {}
    for image_id, group in df.groupby("image_id"):
        boxes_by_image[image_id] = list(group["box"])
    return boxes_by_image


def image_path(image_id):
    return IMAGES_DIR / f"{image_id}.jpg"


def read_image(path):
    """cv2.imread falla con rutas no-ASCII en Windows (p.ej. 'º'); rodeamos vía numpy."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path, image):
    ext = Path(path).suffix
    success, encoded = cv2.imencode(ext, image)
    if not success:
        raise IOError(f"No se pudo codificar la imagen para {path}")
    encoded.tofile(str(path))


def get_split():
    """Split train/val a nivel de imagen, cacheado en results/split.json."""
    if SPLIT_PATH.exists():
        with open(SPLIT_PATH) as f:
            return json.load(f)

    boxes_by_image = load_annotations()
    image_ids = sorted(boxes_by_image.keys())
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(image_ids)

    n_val = max(1, round(len(image_ids) * VAL_FRACTION))
    val_ids = sorted(image_ids[:n_val])
    train_ids = sorted(image_ids[n_val:])

    split = {"train": train_ids, "val": val_ids}
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_PATH, "w") as f:
        json.dump(split, f, indent=2)
    return split


def get_extras():
    return sorted(p.stem for p in EXTRAS_DIR.glob("*.jpg"))
