"""Deteccion de tanques con cv2.HoughCircles + grid search de hiperparametros."""
import itertools

import cv2
import numpy as np

from src import dataset, evaluate

# Radios reales de las anotaciones: entre ~1.5 y ~57 px (ver stats de w/h en annotations.csv)
MIN_RADIUS = 4
MAX_RADIUS = 60

# Subconjunto de imagenes de train usado para el grid search (evita tunear sobre las 78
# imagenes completas, que ya es suficiente para elegir hiperparametros estables)
TUNING_SAMPLE_SIZE = 15

# limite de seguridad: evita que un config demasiado permisivo dispare el coste
# O(n_pred*n_gt) del matching greedy en evaluate.py
MAX_CIRCLES = 3000

PARAM_GRID = {
    "dp": [1.0, 1.5],
    "param1": [120, 150, 180, 200],
    "param2": [70, 80, 90, 100, 110],
    "min_dist_factor": [1.0, 1.5],  # multiplicado por min_radius
}


def preprocess(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.medianBlur(enhanced, 5)
    return blurred


def detect_circles(image_bgr, dp, param1, param2, min_dist_factor):
    preprocessed = preprocess(image_bgr)
    min_dist = max(1, int(MIN_RADIUS * min_dist_factor))
    circles = cv2.HoughCircles(
        preprocessed,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=MIN_RADIUS,
        maxRadius=MAX_RADIUS,
    )
    if circles is None:
        return []
    circles = np.round(circles[0]).astype(int)
    # salvaguarda: params demasiado permisivos pueden disparar decenas de miles de
    # falsos circulos, lo que vuelve el matching O(n_pred*n_gt) impracticable
    if len(circles) > MAX_CIRCLES:
        circles = circles[:MAX_CIRCLES]
    boxes = [(x - r, y - r, x + r, y + r) for x, y, r in circles]
    return boxes


def _score_config(image_ids, boxes_by_image, config):
    per_image_results = []
    for image_id in image_ids:
        image = dataset.read_image(dataset.image_path(image_id))
        pred_boxes = detect_circles(image, **config)
        gt_boxes = boxes_by_image.get(image_id, [])
        per_image_results.append(evaluate.match_detections(pred_boxes, gt_boxes))
    return evaluate.aggregate_metrics(per_image_results)


def tune_hyperparameters(train_ids, boxes_by_image, sample_size=TUNING_SAMPLE_SIZE, verbose=True):
    """Grid search sobre un subconjunto de train, maximizando F1."""
    sample_ids = train_ids[:sample_size]
    keys = list(PARAM_GRID.keys())
    best_config = None
    best_f1 = -1.0

    for values in itertools.product(*(PARAM_GRID[k] for k in keys)):
        config = dict(zip(keys, values))
        metrics = _score_config(sample_ids, boxes_by_image, config)
        if verbose:
            print(f"  {config} -> F1={metrics['f1']:.3f} P={metrics['precision']:.3f} R={metrics['recall']:.3f}")
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_config = config

    return best_config, best_f1
