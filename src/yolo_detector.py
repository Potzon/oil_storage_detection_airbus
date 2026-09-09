"""Entrenamiento e inferencia por tiles de YOLOv8 sobre imagenes completas."""
from pathlib import Path

from ultralytics import YOLO

from src.evaluate import non_max_suppression
from src.tiling import tile_origins
from src.yolo_data import TILE_SIZE, TILE_STRIDE

DEFAULT_WEIGHTS = "yolov8n.pt"


def train(data_yaml_path, epochs=60, project_dir=None, run_name="oil_tanks"):
    model = YOLO(DEFAULT_WEIGHTS)
    project_dir = str(project_dir) if project_dir else None
    model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=TILE_SIZE,
        project=project_dir,
        name=run_name,
        exist_ok=True,
    )
    best_weights = Path(model.trainer.save_dir) / "weights" / "best.pt"
    return best_weights


def load_model(weights_path):
    return YOLO(str(weights_path))


def detect_full_image(model, image_bgr, conf=0.25, iou_nms=0.4, tile_size=TILE_SIZE, stride=TILE_STRIDE):
    """Corre el modelo por tiles sobre una imagen completa y fusiona con NMS."""
    h, w = image_bgr.shape[:2]
    xs = tile_origins(w, tile_size, stride)
    ys = tile_origins(h, tile_size, stride)

    all_boxes = []
    all_scores = []

    for ty in ys:
        for tx in xs:
            tile = image_bgr[ty:ty + tile_size, tx:tx + tile_size]
            result = model.predict(tile, conf=conf, verbose=False)[0]
            for box, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = box
                all_boxes.append((x1 + tx, y1 + ty, x2 + tx, y2 + ty))
                all_scores.append(float(score))

    if not all_boxes:
        return [], []

    return non_max_suppression(all_boxes, all_scores, iou_threshold=iou_nms)
