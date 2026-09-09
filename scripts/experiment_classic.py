"""Prueba variantes deterministas adicionales (Hough tileado, blob detector, contornos
+circularidad) para ver si alguna supera al Hough Circles de imagen completa (F1=0.063).

Cada variante se tunea con grid search sobre una muestra de train y se evalua sobre el
mismo split de val que ya usan run_hough.py / run_yolo.py. Guarda resultados crudos en
results/classic_experiments.json y un resumen legible en EXPERIMENTS.md.
"""
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import classic_methods as cm
from src import dataset, evaluate

RESULTS_DIR = ROOT / "results"
TUNING_SAMPLE_SIZE = 8

METHODS = {
    "hough_tiled": {
        "fn": cm.hough_tiled_detect,
        "grid": {
            "dp": [1.0, 1.5],
            "param1": [120, 150, 180],
            "param2": [40, 50, 60, 70],
            "min_dist_factor": [1.0],
        },
    },
    "blob_detector": {
        "fn": cm.blob_detector_detect,
        "grid": {
            "min_area": [15, 30],
            "max_area": [3000],
            "min_circularity": [0.6, 0.75],
            "min_convexity": [0.8],
            "blob_color": [0, 255],
        },
    },
    "contour_circularity": {
        "fn": cm.contour_circularity_detect,
        "grid": {
            "block_size": [15, 25],
            "c_offset": [3, 6],
            "min_area": [15, 30],
            "max_area": [3000],
            "min_circularity": [0.5, 0.65],
        },
    },
}


def _score_config(fn, image_ids, boxes_by_image, config):
    per_image_results = []
    for image_id in image_ids:
        image = dataset.read_image(dataset.image_path(image_id))
        pred_boxes = fn(image, **config)
        gt_boxes = boxes_by_image.get(image_id, [])
        per_image_results.append(evaluate.match_detections(pred_boxes, gt_boxes))
    return evaluate.aggregate_metrics(per_image_results)


def tune_method(name, spec, tuning_ids, boxes_by_image):
    keys = list(spec["grid"].keys())
    combos = list(itertools.product(*(spec["grid"][k] for k in keys)))
    print(f"\n[{name}] grid search sobre {len(combos)} configs x {len(tuning_ids)} imagenes...")

    best_config, best_f1 = None, -1.0
    for values in combos:
        config = dict(zip(keys, values))
        metrics = _score_config(spec["fn"], tuning_ids, boxes_by_image, config)
        print(f"  {config} -> F1={metrics['f1']:.3f} P={metrics['precision']:.3f} R={metrics['recall']:.3f}")
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_config = config

    return best_config, best_f1


def evaluate_on_val(name, spec, best_config, val_ids, boxes_by_image):
    per_image_results = []
    times = []
    for image_id in val_ids:
        image = dataset.read_image(dataset.image_path(image_id))
        gt_boxes = boxes_by_image.get(image_id, [])

        t0 = time.perf_counter()
        pred_boxes = spec["fn"](image, **best_config)
        times.append(time.perf_counter() - t0)

        per_image_results.append(evaluate.match_detections(pred_boxes, gt_boxes))

    metrics = evaluate.aggregate_metrics(per_image_results)
    metrics["mean_inference_time_s"] = sum(times) / len(times)
    metrics["best_config"] = best_config
    return metrics


def main():
    boxes_by_image = dataset.load_annotations()
    split = dataset.get_split()
    tuning_ids = split["train"][:TUNING_SAMPLE_SIZE]

    all_results = {}
    for name, spec in METHODS.items():
        best_config, best_f1_train = tune_method(name, spec, tuning_ids, boxes_by_image)
        print(f"[{name}] mejor config: {best_config} (F1 tuning={best_f1_train:.3f})")

        val_metrics = evaluate_on_val(name, spec, best_config, split["val"], boxes_by_image)
        print(f"[{name}] metricas en val: {val_metrics}")
        all_results[name] = val_metrics

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "classic_experiments.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nGuardado en {out_path}")


if __name__ == "__main__":
    main()
