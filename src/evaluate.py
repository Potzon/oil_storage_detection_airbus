"""Matching greedy por IoU, metricas de deteccion (precision/recall/F1) y NMS generico."""
import numpy as np


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def match_detections(pred_boxes, gt_boxes, iou_threshold=0.5, scores=None):
    """Empareja predicciones con ground truth de forma greedy.

    Si se pasan `scores`, las predicciones se procesan de mayor a menor
    confianza. Sin scores (p.ej. Hough) se procesan en el orden recibido.

    Devuelve (matched_ious, n_tp, n_fp, n_fn).
    """
    order = range(len(pred_boxes))
    if scores is not None:
        order = sorted(order, key=lambda i: scores[i], reverse=True)

    gt_used = [False] * len(gt_boxes)
    matched_ious = []
    n_tp = 0

    for i in order:
        best_iou = 0.0
        best_j = -1
        for j, gt_box in enumerate(gt_boxes):
            if gt_used[j]:
                continue
            cur_iou = iou(pred_boxes[i], gt_box)
            if cur_iou > best_iou:
                best_iou = cur_iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_threshold:
            gt_used[best_j] = True
            matched_ious.append(best_iou)
            n_tp += 1

    n_fp = len(pred_boxes) - n_tp
    n_fn = len(gt_boxes) - n_tp
    return matched_ious, n_tp, n_fp, n_fn


def aggregate_metrics(per_image_results):
    """per_image_results: lista de (matched_ious, n_tp, n_fp, n_fn)."""
    total_tp = sum(r[1] for r in per_image_results)
    total_fp = sum(r[2] for r in per_image_results)
    total_fn = sum(r[3] for r in per_image_results)
    all_ious = [iou_val for r in per_image_results for iou_val in r[0]]

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_iou_matched": float(mean_iou),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
    }


def non_max_suppression(boxes, scores, iou_threshold=0.4):
    """NMS generico: boxes es una lista de (x1,y1,x2,y2), scores paralelos.

    Si no hay score real (p.ej. detectores clasicos sin confianza), pasar una
    lista de 1.0 del mismo tamano: el orden de desempate sera el de entrada.
    """
    if not boxes:
        return [], []

    boxes_arr = np.array(boxes, dtype=np.float32)
    scores_arr = np.array(scores, dtype=np.float32)

    x1, y1, x2, y2 = boxes_arr[:, 0], boxes_arr[:, 1], boxes_arr[:, 2], boxes_arr[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores_arr.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        union = areas[i] + areas[order[1:]] - inter
        iou_vals = np.where(union > 0, inter / union, 0.0)

        order = order[1:][iou_vals <= iou_threshold]

    kept_boxes = [tuple(float(v) for v in boxes_arr[i]) for i in keep]
    kept_scores = [float(scores_arr[i]) for i in keep]
    return kept_boxes, kept_scores
