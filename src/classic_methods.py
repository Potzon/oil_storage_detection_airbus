"""Variantes deterministas adicionales para detectar tanques, todas operando por tiles
(igual que YOLO) en vez de sobre la imagen 2560x2560 completa: al reducir el area de
trabajo, el acumulador/threshold local tiene que lidiar con menos tanques agrupados
a la vez, lo que en teoria deberia ayudar a separar clusters densos.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import blob_dog, peak_local_max
from skimage.measure import regionprops

from src.evaluate import non_max_suppression
from src.hough_detector import MAX_RADIUS, MIN_RADIUS, preprocess
from src.tiling import iter_tiles

TILE_SIZE = 640
TILE_STRIDE = 512

# salvaguarda compartida contra explosiones combinatorias de un config demasiado
# permisivo (ver nota en hough_detector.py)
MAX_DETECTIONS_PER_TILE = 500


# ---------------------------------------------------------------------------
# Variante A: Hough Circles, pero corrido tile a tile en vez de imagen completa
# ---------------------------------------------------------------------------

def hough_tiled_detect(image_bgr, dp, param1, param2, min_dist_factor,
                        tile_size=TILE_SIZE, stride=TILE_STRIDE, nms_iou=0.3):
    all_boxes = []
    for tx, ty, tile in iter_tiles(image_bgr, tile_size, stride):
        pre = preprocess(tile)
        min_dist = max(1, int(MIN_RADIUS * min_dist_factor))
        circles = cv2.HoughCircles(
            pre, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
            param1=param1, param2=param2, minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS,
        )
        if circles is None:
            continue
        circles = np.round(circles[0]).astype(int)[:MAX_DETECTIONS_PER_TILE]
        for x, y, r in circles:
            all_boxes.append((x - r + tx, y - r + ty, x + r + tx, y + r + ty))

    if not all_boxes:
        return []
    scores = [1.0] * len(all_boxes)
    boxes, _ = non_max_suppression(all_boxes, scores, iou_threshold=nms_iou)
    return boxes


# ---------------------------------------------------------------------------
# Variante B: cv2.SimpleBlobDetector sobre blobs circulares brillantes/oscuros
# ---------------------------------------------------------------------------

def _build_blob_detector(min_area, max_area, min_circularity, min_convexity, blob_color):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = max_area
    params.filterByCircularity = True
    params.minCircularity = min_circularity
    params.filterByConvexity = True
    params.minConvexity = min_convexity
    params.filterByInertia = False
    params.filterByColor = True
    params.blobColor = blob_color
    params.minThreshold = 10
    params.maxThreshold = 220
    params.thresholdStep = 10
    return cv2.SimpleBlobDetector_create(params)


def blob_detector_detect(image_bgr, min_area, max_area, min_circularity, min_convexity,
                          blob_color=255, tile_size=TILE_SIZE, stride=TILE_STRIDE, nms_iou=0.3):
    detector = _build_blob_detector(min_area, max_area, min_circularity, min_convexity, blob_color)
    all_boxes = []
    for tx, ty, tile in iter_tiles(image_bgr, tile_size, stride):
        gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        keypoints = detector.detect(enhanced)[:MAX_DETECTIONS_PER_TILE]
        for kp in keypoints:
            x, y = kp.pt
            r = kp.size / 2
            all_boxes.append((x - r + tx, y - r + ty, x + r + tx, y + r + ty))

    if not all_boxes:
        return []
    scores = [1.0] * len(all_boxes)
    boxes, _ = non_max_suppression(all_boxes, scores, iou_threshold=nms_iou)
    return boxes


# ---------------------------------------------------------------------------
# Variante C: umbral adaptativo + contornos filtrados por circularidad, con
# separacion por watershed cuando varios tanques quedan fusionados en un blob
# ---------------------------------------------------------------------------

def _split_merged_blob(mask, min_area):
    """Separa un blob grande (varios tanques pegados) via distance transform + watershed."""
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return []
    _, peaks = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
    peaks = peaks.astype(np.uint8)
    n_markers, markers = cv2.connectedComponents(peaks)
    if n_markers <= 2:
        return []

    markers = markers + 1
    markers[mask == 0] = 0
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.watershed(mask_bgr, markers)

    boxes = []
    for label in range(2, n_markers + 1):
        region = np.uint8(markers == label) * 255
        if cv2.countNonZero(region) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(region)
        boxes.append((x, y, x + w, y + h))
    return boxes


def contour_circularity_detect(image_bgr, block_size, c_offset, min_area, max_area,
                                min_circularity, tile_size=TILE_SIZE, stride=TILE_STRIDE,
                                nms_iou=0.3, split_merged=True):
    all_boxes = []
    for tx, ty, tile in iter_tiles(image_bgr, tile_size, stride):
        gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            block_size, -c_offset,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tile_boxes = []
        for cnt in contours[:MAX_DETECTIONS_PER_TILE]:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

            if area > max_area and split_merged:
                blob_mask = np.zeros_like(binary)
                cv2.drawContours(blob_mask, [cnt], -1, 255, -1)
                tile_boxes.extend(_split_merged_blob(blob_mask, min_area))
                continue

            if circularity < min_circularity or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            tile_boxes.append((x, y, x + w, y + h))

        for x1, y1, x2, y2 in tile_boxes:
            all_boxes.append((x1 + tx, y1 + ty, x2 + tx, y2 + ty))

    if not all_boxes:
        return []
    scores = [1.0] * len(all_boxes)
    boxes, _ = non_max_suppression(all_boxes, scores, iou_threshold=nms_iou)
    return boxes


# ---------------------------------------------------------------------------
# Variante D: watershed marker-controlled aplicado a TODA la mascara de foreground,
# no solo a los contornos ya sospechosos de estar fusionados (a diferencia de la
# variante C). Es el algoritmo clasico de libro para separar objetos circulares
# que se tocan (el mismo que se usa para separar celulas pegadas en microscopia):
# distance transform + maximos locales como semillas -> watershed global.
# ---------------------------------------------------------------------------

def watershed_universal_detect(image_bgr, brightness_percentile, min_distance, min_area,
                                max_area, min_circularity, tile_size=TILE_SIZE,
                                stride=TILE_STRIDE, nms_iou=0.3):
    """Nota: la primera version usaba adaptiveThreshold (igual que la variante C), pero
    generaba una mascara de foreground demasiado extensa (30-40% de la imagen: cualquier
    zona algo mas clara que su entorno local, incluyendo carreteras y campos), lo que
    hacia que el distance-transform tuviera miles de maximos locales espurios y el
    watershed sobre-segmentara todo en fragmentos de ~1px. Un umbral global por percentil
    de brillo (top N% mas claro de la escena) aisla mucho mejor los techos de los tanques,
    que son el objeto mas brillante y compacto de la imagen.
    """
    all_boxes = []
    for tx, ty, tile in iter_tiles(image_bgr, tile_size, stride):
        gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

        thresh_val = np.percentile(blurred, brightness_percentile)
        binary = (blurred > thresh_val).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        if cv2.countNonZero(binary) == 0:
            continue

        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        peak_coords = peak_local_max(dist, min_distance=min_distance, labels=binary.astype(bool))
        if len(peak_coords) == 0:
            continue

        # receta estandar de watershed marker-controlled: hay que decirle explicitamente
        # donde esta el FONDO seguro (si no, cv2.watershed no tiene semilla desde la que
        # crecer el fondo y deja casi todo como frontera/sin asignar, ver EXPERIMENTS.md).
        peak_mask = np.zeros_like(binary, dtype=bool)
        peak_mask[tuple(peak_coords.T)] = True
        sure_fg = cv2.dilate(peak_mask.astype(np.uint8) * 255, kernel, iterations=1)
        sure_bg = cv2.dilate(binary, kernel, iterations=3)
        unknown = cv2.subtract(sure_bg, sure_fg)

        n_markers, markers = cv2.connectedComponents(sure_fg)
        if n_markers <= 1:
            continue
        markers = markers + 1
        markers[unknown > 0] = 0
        watershed_labels = cv2.watershed(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB).copy(), markers.astype(np.int32))

        # un solo pase con regionprops en vez de reconstruir una mascara 640x640 por
        # cada label (con miles de semillas eso hacia el metodo O(n_labels*area_tile))
        watershed_labels = np.clip(watershed_labels, 0, None)  # bordes de watershed valen -1
        for prop in regionprops(watershed_labels):
            if prop.label <= 1:  # 0=fondo sin asignar, 1=marcador de fondo
                continue
            area = prop.area
            if area < min_area or area > max_area:
                continue
            perimeter = prop.perimeter
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            if circularity < min_circularity:
                continue
            minr, minc, maxr, maxc = prop.bbox
            all_boxes.append((minc + tx, minr + ty, maxc + tx, maxr + ty))

    if not all_boxes:
        return []
    scores = [1.0] * len(all_boxes)
    boxes, _ = non_max_suppression(all_boxes, scores, iou_threshold=nms_iou)
    return boxes


# ---------------------------------------------------------------------------
# Variante E: deteccion multiescala Difference-of-Gaussians (skimage.feature.blob_dog),
# una aproximacion mucho mas rapida de Laplacian-of-Gaussian (blob_log tardaba ~3s/tile,
# DoG ~0.1-0.5s/tile). A diferencia de Hough/SimpleBlobDetector (escala fija por config),
# DoG barre un rango de sigmas y busca maximos en el espacio escala-posicion, lo que en
# teoria deberia manejar mejor el rango real de radios del dataset (3-108px).
# ---------------------------------------------------------------------------

def dog_blob_detect(image_bgr, min_sigma, max_sigma, sigma_ratio, threshold,
                     tile_size=TILE_SIZE, stride=TILE_STRIDE, nms_iou=0.3):
    all_boxes = []
    for tx, ty, tile in iter_tiles(image_bgr, tile_size, stride):
        gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray).astype(np.float64) / 255.0

        blobs = blob_dog(
            enhanced, min_sigma=min_sigma, max_sigma=max_sigma,
            sigma_ratio=sigma_ratio, threshold=threshold,
        )
        if len(blobs) > MAX_DETECTIONS_PER_TILE:
            blobs = blobs[:MAX_DETECTIONS_PER_TILE]

        for y, x, sigma in blobs:
            r = sigma * np.sqrt(2)
            all_boxes.append((x - r + tx, y - r + ty, x + r + tx, y + r + ty))

    if not all_boxes:
        return []
    scores = [1.0] * len(all_boxes)
    boxes, _ = non_max_suppression(all_boxes, scores, iou_threshold=nms_iou)
    return boxes


# ---------------------------------------------------------------------------
# Post-proceso: filtro por densidad de vecinos. Los tanques reales casi nunca aparecen
# aislados -- se instalan en "granjas de tanques" (clusters muy densos, ver stats del
# dataset: mediana ~100 tanques/imagen). Una deteccion sin ningun otro tanque candidato
# cerca es, estadisticamente, mas probable que sea ruido (un tejado, una piscina, un
# parche de terreno) que un tanque real. Este filtro es agnostico al detector: se aplica
# como post-proceso sobre las cajas de cualquiera de las variantes de arriba.
# ---------------------------------------------------------------------------

def filter_by_density(boxes, radius, min_neighbors):
    """Descarta cajas sin al menos `min_neighbors` otras cajas dentro de `radius` px
    (distancia entre centros)."""
    if not boxes:
        return boxes

    centers = np.array([((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in boxes])
    kept = []
    for i, center in enumerate(centers):
        dists = np.linalg.norm(centers - center, axis=1)
        n_neighbors = np.sum((dists > 0) & (dists <= radius))
        if n_neighbors >= min_neighbors:
            kept.append(boxes[i])
    return kept
