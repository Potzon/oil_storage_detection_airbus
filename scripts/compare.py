"""Compara Hough Circles vs YOLOv8: tabla de metricas + visualizaciones conjuntas."""
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import dataset, hough_detector, visualize, yolo_detector

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DEFAULT_YOLO_WEIGHTS = RESULTS_DIR / "yolo_runs" / "oil_tanks" / "weights" / "best.pt"

N_QUALITATIVE_VAL_IMAGES = 3


def print_comparison_table(hough_metrics, yolo_metrics):
    rows = ["precision", "recall", "f1", "mean_iou_matched", "mean_inference_time_s"]
    header = f"{'metrica':<24}{'Hough (clasico)':<20}{'YOLOv8 (CNN)':<20}"
    print(header)
    print("-" * len(header))
    for row in rows:
        h_val = hough_metrics.get(row, float("nan"))
        y_val = yolo_metrics.get(row, float("nan"))
        print(f"{row:<24}{h_val:<20.4f}{y_val:<20.4f}")


def main():
    hough_metrics_path = RESULTS_DIR / "hough_metrics.json"
    yolo_metrics_path = RESULTS_DIR / "yolo_metrics.json"

    if not hough_metrics_path.exists() or not yolo_metrics_path.exists():
        raise SystemExit(
            "Faltan resultados: ejecuta antes scripts/run_hough.py y scripts/run_yolo.py"
        )

    with open(hough_metrics_path) as f:
        hough_metrics = json.load(f)
    with open(yolo_metrics_path) as f:
        yolo_metrics = json.load(f)

    print_comparison_table(hough_metrics, yolo_metrics)

    combined = {"hough": hough_metrics, "yolo": yolo_metrics}
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(combined, f, indent=2)

    # visualizaciones lado a lado (GT + Hough + YOLO) sobre unas cuantas imagenes de val
    boxes_by_image = dataset.load_annotations()
    split = dataset.get_split()
    best_hough_config = hough_metrics["best_config"]
    model = yolo_detector.load_model(DEFAULT_YOLO_WEIGHTS)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for image_id in split["val"][:N_QUALITATIVE_VAL_IMAGES]:
        image = dataset.read_image(dataset.image_path(image_id))
        gt_boxes = boxes_by_image.get(image_id, [])
        hough_boxes = hough_detector.detect_circles(image, **best_hough_config)
        yolo_boxes, _ = yolo_detector.detect_full_image(model, image)

        vis = visualize.render_comparison(
            image, gt_boxes=gt_boxes, hough_boxes=hough_boxes, yolo_boxes=yolo_boxes
        )
        dataset.write_image(FIGURES_DIR / f"compare_{image_id}.jpg", vis)

    # inferencia cualitativa sobre las imagenes extra (sin anotar)
    for extra_id in dataset.get_extras():
        image = dataset.read_image(dataset.EXTRAS_DIR / f"{extra_id}.jpg")
        hough_boxes = hough_detector.detect_circles(image, **best_hough_config)
        yolo_boxes, _ = yolo_detector.detect_full_image(model, image)

        vis = visualize.render_comparison(image, hough_boxes=hough_boxes, yolo_boxes=yolo_boxes)
        dataset.write_image(FIGURES_DIR / f"extra_{extra_id}.jpg", vis)

    print(f"\nVisualizaciones guardadas en {FIGURES_DIR}")


if __name__ == "__main__":
    main()
