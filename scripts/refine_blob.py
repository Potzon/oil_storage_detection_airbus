"""Segunda pasada de grid search, mas fina, alrededor de la mejor region encontrada
para blob_detector en experiment_classic.py (el metodo clasico mas prometedor)."""
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
TUNING_SAMPLE_SIZE = 12

GRID = {
    "min_area": [20, 30, 40],
    "max_area": [3000],
    "min_circularity": [0.7, 0.8, 0.85],
    "min_convexity": [0.75, 0.85, 0.9],
    "blob_color": [255],
}


def _score_config(image_ids, boxes_by_image, config):
    per_image_results = []
    for image_id in image_ids:
        image = dataset.read_image(dataset.image_path(image_id))
        pred_boxes = cm.blob_detector_detect(image, **config)
        gt_boxes = boxes_by_image.get(image_id, [])
        per_image_results.append(evaluate.match_detections(pred_boxes, gt_boxes))
    return evaluate.aggregate_metrics(per_image_results)


def main():
    boxes_by_image = dataset.load_annotations()
    split = dataset.get_split()
    tuning_ids = split["train"][:TUNING_SAMPLE_SIZE]

    keys = list(GRID.keys())
    combos = list(itertools.product(*(GRID[k] for k in keys)))
    print(f"grid search fino sobre {len(combos)} configs x {len(tuning_ids)} imagenes...")

    best_config, best_f1 = None, -1.0
    for values in combos:
        config = dict(zip(keys, values))
        metrics = _score_config(tuning_ids, boxes_by_image, config)
        print(f"  {config} -> F1={metrics['f1']:.3f} P={metrics['precision']:.3f} R={metrics['recall']:.3f}")
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_config = config

    print(f"\nMejor config fino: {best_config} (F1 tuning={best_f1:.3f})")

    per_image_results = []
    times = []
    for image_id in split["val"]:
        image = dataset.read_image(dataset.image_path(image_id))
        gt_boxes = boxes_by_image.get(image_id, [])
        t0 = time.perf_counter()
        pred_boxes = cm.blob_detector_detect(image, **best_config)
        times.append(time.perf_counter() - t0)
        per_image_results.append(evaluate.match_detections(pred_boxes, gt_boxes))

    metrics = evaluate.aggregate_metrics(per_image_results)
    metrics["mean_inference_time_s"] = sum(times) / len(times)
    metrics["best_config"] = best_config
    print(f"\nMetricas en val (refinado): {metrics}")

    out_path = RESULTS_DIR / "blob_detector_refined.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nGuardado en {out_path}")


if __name__ == "__main__":
    main()
