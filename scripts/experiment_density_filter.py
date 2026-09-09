"""Prueba un post-proceso de filtro por densidad de vecinos sobre el mejor detector
determinista hasta ahora (blob detector refinado, F1=0.354 en val). Hipotesis: los
tanques reales aparecen casi siempre en clusters, asi que una deteccion aislada (sin
otras cajas cerca) es probablemente ruido y se puede descartar sin perder apenas recall.

Las cajas del blob detector se calculan una sola vez por imagen (es la parte cara) y se
reutilizan para probar todas las combinaciones de (radius, min_neighbors) del filtro
(que es casi gratis), tanto en tuning como en la evaluacion final de val.
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
TUNING_SAMPLE_SIZE = 12

FILTER_GRID = {
    "radius": [100, 150, 200, 250, 300],
    "min_neighbors": [1, 2, 3],
}


def _load_best_blob_config():
    with open(RESULTS_DIR / "blob_detector_refined.json") as f:
        return json.load(f)["best_config"]


def _compute_raw_boxes(image_ids, blob_config):
    """Corre el blob detector una sola vez por imagen y cachea las cajas."""
    boxes_by_id = {}
    times = []
    for image_id in image_ids:
        image = dataset.read_image(dataset.image_path(image_id))
        t0 = time.perf_counter()
        boxes_by_id[image_id] = cm.blob_detector_detect(image, **blob_config)
        times.append(time.perf_counter() - t0)
    return boxes_by_id, sum(times) / len(times)


def main():
    boxes_by_image = dataset.load_annotations()
    split = dataset.get_split()
    blob_config = _load_best_blob_config()
    print(f"Config base (blob detector refinado): {blob_config}")

    tuning_ids = split["train"][:TUNING_SAMPLE_SIZE]
    print(f"\nCalculando cajas base sobre {len(tuning_ids)} imagenes de tuning...")
    tuning_raw_boxes, _ = _compute_raw_boxes(tuning_ids, blob_config)

    keys = list(FILTER_GRID.keys())
    combos = list(itertools.product(*(FILTER_GRID[k] for k in keys)))
    print(f"\nGrid search del filtro de densidad sobre {len(combos)} configs (cajas base ya calculadas)...")

    best_filter_config, best_f1 = None, -1.0
    baseline_metrics = evaluate.aggregate_metrics(
        [evaluate.match_detections(tuning_raw_boxes[i], boxes_by_image.get(i, [])) for i in tuning_ids]
    )
    print(f"  [sin filtro] F1={baseline_metrics['f1']:.3f} P={baseline_metrics['precision']:.3f} R={baseline_metrics['recall']:.3f}")

    for values in combos:
        filter_config = dict(zip(keys, values))
        per_image_results = []
        for image_id in tuning_ids:
            filtered = cm.filter_by_density(tuning_raw_boxes[image_id], **filter_config)
            gt_boxes = boxes_by_image.get(image_id, [])
            per_image_results.append(evaluate.match_detections(filtered, gt_boxes))
        metrics = evaluate.aggregate_metrics(per_image_results)
        print(f"  {filter_config} -> F1={metrics['f1']:.3f} P={metrics['precision']:.3f} R={metrics['recall']:.3f}")
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_filter_config = filter_config

    print(f"\nMejor filtro: {best_filter_config} (F1 tuning={best_f1:.3f} vs sin filtro={baseline_metrics['f1']:.3f})")

    print(f"\nEvaluando en val (20 imagenes)...")
    val_raw_boxes, mean_blob_time = _compute_raw_boxes(split["val"], blob_config)

    per_image_results_nofilter = []
    per_image_results_filtered = []
    filter_times = []
    for image_id in split["val"]:
        gt_boxes = boxes_by_image.get(image_id, [])
        raw = val_raw_boxes[image_id]
        per_image_results_nofilter.append(evaluate.match_detections(raw, gt_boxes))

        t0 = time.perf_counter()
        filtered = cm.filter_by_density(raw, **best_filter_config)
        filter_times.append(time.perf_counter() - t0)
        per_image_results_filtered.append(evaluate.match_detections(filtered, gt_boxes))

    metrics_nofilter = evaluate.aggregate_metrics(per_image_results_nofilter)
    metrics_filtered = evaluate.aggregate_metrics(per_image_results_filtered)
    metrics_filtered["mean_inference_time_s"] = mean_blob_time + sum(filter_times) / len(filter_times)
    metrics_filtered["best_filter_config"] = best_filter_config
    metrics_filtered["blob_config"] = blob_config

    print(f"\n[sin filtro]  P={metrics_nofilter['precision']:.3f} R={metrics_nofilter['recall']:.3f} F1={metrics_nofilter['f1']:.3f}")
    print(f"[con filtro]  P={metrics_filtered['precision']:.3f} R={metrics_filtered['recall']:.3f} F1={metrics_filtered['f1']:.3f}")

    out_path = RESULTS_DIR / "density_filter.json"
    with open(out_path, "w") as f:
        json.dump({"no_filter": metrics_nofilter, "with_filter": metrics_filtered}, f, indent=2)
    print(f"\nGuardado en {out_path}")


if __name__ == "__main__":
    main()
