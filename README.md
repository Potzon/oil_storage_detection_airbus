# Detección de tanques de petróleo en imágenes satelitales

**OpenCV clásico vs. YOLOv8** · Proyecto individual de *Fundamentos de la Visión por Computador*

Detección de tanques de almacenamiento de petróleo (círculos casi perfectos, agrupados en
"granjas" de decenas o cientos de unidades) en imágenes de satélite de 2560×2560 px. El
proyecto compara **7 técnicas deterministas** (sin aprendizaje) frente a una **CNN
fine-tuned (YOLOv8n)**, todas bajo exactamente el mismo protocolo de evaluación.

## Resultados principales

Evaluación sobre 20 imágenes de validación (IoU ≥ 0.5):

| Método | Precisión | Recall | F1 | Tiempo/img |
|---|---|---|---|---|
| Hough Circles, imagen completa | 0.113 | 0.044 | 0.063 | 0.073 s |
| DoG multiescala | 0.031 | 0.052 | 0.039 | 3.363 s |
| Contornos + circularidad + watershed local | 0.144 | 0.107 | 0.123 | 0.193 s |
| Hough Circles, tileado | 0.443 | 0.104 | 0.169 | 0.175 s |
| Watershed universal | 0.287 | 0.217 | 0.247 | 0.794 s |
| Blob detector (grid inicial) | 0.323 | 0.291 | 0.306 | 0.439 s |
| **Blob detector (grid refinado)** | 0.584 | 0.254 | **0.354** | 0.429 s |
| Blob detector + filtro de densidad | 0.626 | 0.248 | 0.355 | 0.428 s |
| **YOLOv8n fine-tuned** | **0.812** | **0.876** | **0.843** | 0.373 s |

**Conclusiones clave**

- El techo del enfoque clásico está en **F1 ≈ 0.35**, menos de la mitad que YOLOv8n (0.843).
- El cuello de botella es el **recall**, no la geometría: cuando un método clásico acierta,
  la caja es tan buena como la de la CNN (IoU medio 0.56–0.81).
- **Elegir bien la técnica importa más que afinarla**: pasar de Hough a `SimpleBlobDetector`
  duplicó el F1. Las técnicas que filtran por **polaridad de brillo** (tanques = objetos
  claros y compactos) superan con margen a las que solo miran bordes.
- Un post-proceso barato (filtro por densidad de vecinos) **no rompe el techo**: los falsos
  positivos también aparecen en grupo (naves industriales, urbanizaciones).

## Dataset

[Airbus Oil Storage Detection](https://www.kaggle.com/datasets/airbusgeo/airbus-oil-storage-detection-dataset)
(Airbus DS / Scale AI, licencia CC BY-NC-SA 4.0).

| Propiedad | Valor |
|---|---|
| Imágenes | 98 anotadas (+ 5 sin anotar en `extras/`) |
| Resolución | 2560 × 2560 px (~1.2–1.5 m/px, SPOT) |
| Anotaciones | 13 592 cajas, clase única `oil-storage-tank` |
| Tanques por imagen | mín. 14 · mediana 99.5 · máx. 893 |
| Lado de la caja | mín. 3 px · mediana 18 px · máx. 108 px |

El dataset **no se incluye** en el repositorio; descárgalo de Kaggle y colócalo en
`airbus_oil_storage_detection/`.

## Protocolo de evaluación

Común a todos los métodos, para que la comparación sea honesta:

- **Split fijo 78 train / 20 val** a nivel de imagen (semilla 42, guardado en `results/split.json`).
- **Tuning solo sobre train** por grid search maximizando F1; la configuración se congela y
  se evalúa una única vez sobre val.
- **Matching greedy por IoU ≥ 0.5**, sin duplicados.
- **Tiling** de 640×640 px con 64 px de solape (igual que YOLO) y NMS para fusionar tiles.
- Métricas: precisión, recall, F1, IoU medio de los aciertos y tiempo por imagen.

## Estructura del proyecto

```
.
├── airbus_oil_storage_detection/   # dataset original (imágenes + annotations.csv)
├── src/
│   ├── dataset.py                  # parseo de anotaciones, split train/val, I/O de imagen
│   ├── evaluate.py                 # IoU, matching greedy, métricas P/R/F1, NMS
│   ├── tiling.py                   # partición en tiles con solape
│   ├── hough_detector.py           # Hough Circles (imagen completa) + grid search
│   ├── classic_methods.py          # resto de detectores clásicos + filtro de densidad
│   ├── yolo_data.py                # conversión del dataset a formato YOLO
│   ├── yolo_detector.py            # entrenamiento e inferencia tileada de YOLOv8
│   └── visualize.py                # superposición de cajas GT / predicción
├── scripts/
│   ├── run_hough.py                # Hough sobre imagen completa
│   ├── experiment_classic.py       # Hough tileado, blob detector, contornos
│   ├── refine_blob.py              # tuning fino del blob detector
│   ├── experiment_classic2.py      # watershed universal y DoG
│   ├── experiment_density_filter.py# post-proceso por densidad
│   ├── train_yolo.py               # genera el dataset YOLO y entrena
│   ├── run_yolo.py                 # evalúa YOLO con el mismo protocolo
│   └── compare.py                  # tabla comparativa final + figuras
├── results/                        # métricas (JSON) y figuras
│   └── figures/                    # comparativas visuales GT / blob / YOLO
├── EXPERIMENTS.md                  # bitácora detallada de los experimentos
└── MEMORIA.md                      # memoria completa del proyecto
```

## Técnicas evaluadas

1. **Hough Circles, imagen completa** — `cv2.HoughCircles` tras CLAHE + filtro de mediana.
2. **Hough Circles, tileado** — la misma transformada por ventanas, fusionada con NMS.
3. **Contornos + circularidad + watershed local** — umbral adaptativo; watershed solo en
   contornos sospechosamente grandes.
4. **`cv2.SimpleBlobDetector`** — regiones estables multiumbral filtradas por área,
   circularidad, convexidad y polaridad (`blob_color=255`). *Mejor método clásico.*
5. **Watershed universal controlado por marcadores** — umbral por percentil de brillo,
   distance transform, semillas de primer plano y fondo seguro.
6. **DoG multiescala** — `skimage.feature.blob_dog`.
7. **Filtro por densidad de vecinos** — post-proceso sobre el mejor detector.

Configuración ganadora del blob detector: `min_area=40, min_circularity=0.8,
min_convexity=0.75, blob_color=255`.

Como referencia de aprendizaje profundo se entrena **YOLOv8n** (preentrenado en COCO,
60 épocas, imgsz 640, NVIDIA RTX 3060), con tiles de 640×640 que contengan al menos un
tanque y descartando cajas con menos del 40 % de su área dentro del tile.

## Instalación

Requiere Python 3 y, para entrenar YOLO en local, una GPU con CUDA.

Librerías principales:

```bash
pip install opencv-python numpy pandas scipy scikit-image matplotlib ultralytics
```

## Uso

Los scripts se ejecutan desde la raíz del proyecto. Orden sugerido para reproducir los
resultados:

```bash
# Métodos deterministas
python scripts/run_hough.py
python scripts/experiment_classic.py
python scripts/refine_blob.py
python scripts/experiment_classic2.py
python scripts/experiment_density_filter.py

# Referencia CNN
python scripts/train_yolo.py
python scripts/run_yolo.py

# Tabla comparativa y figuras
python scripts/compare.py
```

Las métricas crudas quedan en `results/*.json` y las visualizaciones en `results/figures/`.

## Documentación

- [`MEMORIA.md`](MEMORIA.md) — memoria completa: problema, estado del arte, metodología,
  resultados, líneas futuras y bibliografía.
- [`EXPERIMENTS.md`](EXPERIMENTS.md) — bitácora experimento a experimento, incluido un bug
  real de diseño en el watershed universal (fondo sin marcar → 99.97 % de regiones de 1 px).

## Líneas de mejora

- Usar **color real** (HSV) en vez de solo brillo en escala de grises.
- **Cajas circulares** explícitas en la red neuronal (Wang et al., 2025).
- **Ensemble** de blob detector + watershed universal con NMS.
- Ajuste fino del lado YOLO: umbral de confianza/NMS, modelos mayores, más épocas, tiles
  más pequeños.
- Ampliar el dataset de entrenamiento (solo hay 78 imágenes de train).

## Referencias principales

- Faudi, J. — *Oil Storage Detection on Airbus Imagery with YOLOX* (AP@0.5 = 0.856).
- Ouzounis, G. — *Oil-Storage Tank Instance Segmentation with Mask R-CNN*.
- Wang et al. (2025) — *Storage tank detection in remote sensing images based on circular
  bounding boxes and large selective kernel*, Scientific Reports.
- Jocher, Chaurasia & Qiu (2023) — *Ultralytics YOLOv8*.

Bibliografía completa en la sección 4 de [`MEMORIA.md`](MEMORIA.md).

## Licencia

El dataset se distribuye bajo **CC BY-NC-SA 4.0** (uso no comercial). Añade aquí la licencia
del código si corresponde.