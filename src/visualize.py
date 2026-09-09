"""Dibuja ground truth y predicciones (Hough / YOLO) sobre una imagen."""
import cv2

GT_COLOR = (0, 200, 0)      # verde (BGR)
HOUGH_COLOR = (255, 128, 0)  # azul
YOLO_COLOR = (0, 0, 255)     # rojo


def draw_boxes(image, boxes, color, thickness=2):
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return image


def render_comparison(image_bgr, gt_boxes=None, hough_boxes=None, yolo_boxes=None):
    """Devuelve una copia de la imagen con las cajas superpuestas y una leyenda."""
    out = image_bgr.copy()
    if gt_boxes:
        draw_boxes(out, gt_boxes, GT_COLOR)
    if hough_boxes:
        draw_boxes(out, hough_boxes, HOUGH_COLOR)
    if yolo_boxes:
        draw_boxes(out, yolo_boxes, YOLO_COLOR)

    legend = [
        ("GT", GT_COLOR, gt_boxes is not None),
        ("Hough", HOUGH_COLOR, hough_boxes is not None),
        ("YOLO", YOLO_COLOR, yolo_boxes is not None),
    ]
    y = 40
    for label, color, active in legend:
        if not active:
            continue
        cv2.putText(out, label, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        y += 40
    return out
