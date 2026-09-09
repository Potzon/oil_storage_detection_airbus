"""Genera el dataset YOLO (tiling) y entrena YOLOv8n sobre la clase oil-storage-tank."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import dataset, yolo_data, yolo_detector

RUNS_DIR = ROOT / "results" / "yolo_runs"


def main():
    boxes_by_image = dataset.load_annotations()
    split = dataset.get_split()

    print("Generando dataset YOLO (tiling 640x640 con overlap)...")
    data_yaml_path = yolo_data.build_yolo_dataset(split["train"], split["val"], boxes_by_image)
    print(f"Dataset YOLO listo en {data_yaml_path}")

    print("\nEntrenando YOLOv8n...")
    best_weights = yolo_detector.train(data_yaml_path, epochs=60, project_dir=RUNS_DIR)
    print(f"\nMejores pesos guardados en: {best_weights}")


if __name__ == "__main__":
    main()
