"""Segunda tanda de variantes deterministas: watershed universal y blob DoG multiescala.
Mismo protocolo que experiment_classic.py, pero en un script separado para no repetir
las 3 variantes ya evaluadas (serian ~20 min perdidos). Fusiona sus resultados en
results/classic_experiments.json en vez de sobreescribirlo.
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
    "watershed_universal": {
        "fn": cm.watershed_universal_detect,
        "grid": {
            "brightness_percentile": [95, 97, 99],
            "min_distance": [5, 8],
            "min_area": [30, 50],
            "max_area": [3000],
            "min_circularity": [0.8, 0.9],
        },
    },
    "dog_blob": {
        "fn": cm.dog_blob_detect,
        "grid": {
            "min_sigma": [1.5, 2.5],
            "max_sigma": [15, 25],
            "sigma_ratio": [1.8, 2.2],
            "threshold": [0.1, 0.15, 0.2],
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
        t0 = time.perf_counter()
        metrics = _score_config(spec["fn"], tuning_ids, boxes_by_image, config)
        dt = time.perf_counter() - t0
        print(f"  {config} -> F1={metrics['f1']:.3f} P={metrics['precision']:.3f} R={metrics['recall']:.3f} ({dt:.1f}s)")
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_config = config

    return best_config, best_f1


def evaluate_on_val(spec, best_config, val_ids, boxes_by_image):
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

    out_path = RESULTS_DIR / "classic_experiments.json"
    all_results = {}
    if out_path.exists():
        with open(out_path) as f:
            all_results = json.load(f)

    for name, spec in METHODS.items():
        best_config, best_f1_train = tune_method(name, spec, tuning_ids, boxes_by_image)
        print(f"[{name}] mejor config: {best_config} (F1 tuning={best_f1_train:.3f})")

        val_metrics = evaluate_on_val(spec, best_config, split["val"], boxes_by_image)
        print(f"[{name}] metricas en val: {val_metrics}")
        all_results[name] = val_metrics

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nGuardado en {out_path}")


if __name__ == "__main__":
    main()
