"""Tunea Hough Circles en train y evalua en val. Guarda un ejemplo visual."""
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import dataset, evaluate, hough_detector, visualize

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def main():
    boxes_by_image = dataset.load_annotations()
    split = dataset.get_split()

    print(f"Train: {len(split['train'])} imagenes | Val: {len(split['val'])} imagenes")
    print("Tuneando hiperparametros de HoughCircles sobre una muestra de train...")
    best_config, best_f1 = hough_detector.tune_hyperparameters(split["train"], boxes_by_image)
    print(f"\nMejor config: {best_config} (F1 train muestra = {best_f1:.3f})")

    print("\nEvaluando en val...")
    per_image_results = []
    times = []
    example_saved = False
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for image_id in split["val"]:
        image = dataset.read_image(dataset.image_path(image_id))
        gt_boxes = boxes_by_image.get(image_id, [])

        t0 = time.perf_counter()
        pred_boxes = hough_detector.detect_circles(image, **best_config)
        times.append(time.perf_counter() - t0)

        result = evaluate.match_detections(pred_boxes, gt_boxes)
        per_image_results.append(result)

        if not example_saved:
            vis = visualize.render_comparison(image, gt_boxes=gt_boxes, hough_boxes=pred_boxes)
            dataset.write_image(FIGURES_DIR / f"hough_example_{image_id}.jpg", vis)
            example_saved = True

    metrics = evaluate.aggregate_metrics(per_image_results)
    metrics["mean_inference_time_s"] = sum(times) / len(times)
    metrics["best_config"] = best_config

    print("\n=== Metricas Hough Circles (val) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "hough_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nGuardado en {out_path}")


if __name__ == "__main__":
    main()
