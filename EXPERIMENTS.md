# Experimentos: mejorando el metodo determinista

Registro de todas las variantes clasicas (sin aprendizaje) probadas para detectar
tanques de petroleo, con el objetivo de ver hasta donde se puede empujar un enfoque
100% basado en OpenCV clasico antes de comparar contra el YOLOv8 fine-tuned.

Protocolo comun a todos los experimentos (ver `src/dataset.py`, `src/evaluate.py`):
- Split fijo 78 train / 20 val a nivel de imagen (`results/split.json`, seed 42).
- Cada metodo se tunea por grid search maximizando F1 sobre una muestra de **train**
  (nunca sobre val) y se congela antes de evaluar en val.
- Matching greedy por IoU >= 0.5 contra las anotaciones, sin duplicados.
- Metrica principal: F1. Se reporta tambien precision, recall, IoU medio de los
  aciertos (mide calidad de localizacion, no de deteccion) y tiempo de inferencia.

## Resumen de resultados (val, 20 imagenes)

| Metodo | Precision | Recall | F1 | IoU medio (aciertos) | Tiempo/img |
|---|---|---|---|---|---|
| Hough Circles, imagen completa | 0.113 | 0.044 | **0.063** | 0.814 | 0.073s |
| DoG blob multiescala (tileado) | 0.031 | 0.052 | **0.039** | 0.565 | 3.363s |
| Contornos + circularidad (+watershed local) | 0.144 | 0.107 | **0.123** | 0.743 | 0.193s |
| Hough Circles, por tiles 640x640 | 0.443 | 0.104 | **0.169** | 0.769 | 0.175s |
| Watershed universal (marker-controlled) | 0.287 | 0.217 | **0.247** | 0.657 | 0.794s |
| Blob detector (grid inicial) | 0.323 | 0.291 | **0.306** | 0.671 | 0.439s |
| **Blob detector (grid refinado)** | **0.584** | 0.254 | **0.354** | 0.672 | 0.429s |
| YOLOv8n fine-tuned (referencia CNN) | 0.812 | 0.876 | **0.843** | 0.809 | 0.373s |

El mejor determinista (blob detector refinado) multiplica por **~5.6x** el F1 del
Hough ingenuo inicial, pero se queda a menos de la mitad del F1 de YOLO.

## Experimento 1 — Hough Circles, imagen completa (baseline inicial)

`scripts/run_hough.py`, config ganadora: `dp=1.5, param1=200, param2=90, min_dist_factor=1.0`.

Primer intento con un grid search demasiado permisivo (`param2` bajo, ~18-26) llego a
disparar **187,000 circulos falsos** en una sola imagen de 2560x2560: la textura del
terreno agricola y las carreteras generan bordes suficientes para que el acumulador de
Hough encuentre "circulos" por todas partes. Eso ademas volvia el matching greedy
(O(n_pred x n_gt)) impracticable (minutos por imagen). Se soluciono subiendo el rango de
`param2` a 70-110 y `param1` a 120-200, lo que redujo las detecciones a un rango
razonable (decenas a cientos por imagen) y hace que el grid search corra en segundos.

**Resultado**: F1=0.063. Cuando SI acierta, la caja es muy precisa (IoU=0.814,
comparable a YOLO), pero el recall es pesimo (4.4%): con hasta 893 tanques por imagen
agrupados en clusters muy densos, cualquier config lo bastante sensible para separar
tanques contiguos genera demasiados falsos positivos en el resto de la imagen.

## Experimento 2 — Hough Circles por tiles (640x640, igual que YOLO)

`src/classic_methods.hough_tiled_detect`. Hipotesis: si se reduce el area de trabajo del
acumulador de Hough al tamano de un tile, hay menos tanques compitiendo por el mismo
espacio de acumulacion y se pueden bajar los umbrales sin que exploten los falsos
positivos.

**Resultado**: F1=0.169 (x2.7 respecto al baseline). La hipotesis se confirma en
parte: la precision sube mucho (0.113 -> 0.443) porque tilear permite usar umbrales mas
laxos de forma local sin arrastrar ruido de toda la imagen. El recall sigue siendo bajo
(10.4%) porque el problema de fondo -- tanques pegados entre si -- no depende del tamano
de la ventana sino de que Hough busca circulos completos y dos circulos que se tocan
interfieren en el gradiente del borde compartido.

## Experimento 3 — Contornos + circularidad + separacion por watershed

`src/classic_methods.contour_circularity_detect`. Umbral adaptativo (Gaussian) sobre la
imagen realzada con CLAHE, filtrado de contornos por circularidad
(`4*pi*area/perimetro^2`), y si un contorno es mucho mayor que un tanque tipico se asume
que son varios tanques fusionados y se separan con distance transform + watershed antes
de aceptar las cajas resultantes.

**Resultado**: F1=0.123. Mejor que Hough de imagen completa pero peor que Hough
tileado. El umbral adaptativo es sensible al contraste local: en zonas con sombras o
terreno claro genera bordes espurios que pasan el filtro de circularidad, y la
separacion por watershed ayuda con blobs grandes pero no compensa esos falsos positivos
de fondo.

## Experimento 4 — Blob detector (`cv2.SimpleBlobDetector`), grid inicial

`src/classic_methods.blob_detector_detect`, tileado igual que los anteriores. Los
tanques son blobs casi circulares y mas claros que el suelo/asfalto de alrededor, que es
justo lo que `SimpleBlobDetector` esta diseñado para encontrar (agrupa regiones estables
a traves de multiples umbrales, en vez de depender de un unico umbral de borde como
Hough).

Se probo `blob_color=0` (blobs oscuros) y `blob_color=255` (blobs claros): los oscuros
fueron consistentemente malisimos (F1 < 0.01 en casi toda la grid, generaban miles de
falsos positivos sobre sombras y asfalto), confirmando que la señal util esta en los
techos claros de los tanques, no en su sombra.

**Resultado** (mejor config: `min_area=30, min_circularity=0.75, blob_color=255`):
F1=0.306, ya mejor que las dos variantes de Hough juntas. Recall notablemente mas alto
(29.1%) que Hough tileado, con precision similar.

## Experimento 5 — Blob detector, grid refinado

Dado que el experimento 4 fue el mas prometedor, se hizo una segunda pasada de grid
search mas fina alrededor de esa region (`min_area` 20-40, `min_circularity` 0.7-0.85,
`min_convexity` 0.75-0.9), con una muestra de tuning mayor (12 imagenes en vez de 8) para
reducir varianza.

**Resultado** (mejor config: `min_area=40, min_circularity=0.8, min_convexity=0.75,
blob_color=255`): F1=0.354, con precision 0.584 (sube mucho al exigir mayor area minima
y circularidad) a costa de bajar el recall a 0.254. `min_convexity` resulto irrelevante
en todo el rango probado -- los blobs candidatos que sobreviven el filtro de circularidad
ya son convexos de por si, asi que ese hiperparametro no aporta nada en este dataset.

## Experimento 6 — Watershed marker-controlled aplicado a toda la mascara

`src/classic_methods.watershed_universal_detect`. Hipotesis: en vez de aplicar
watershed solo quando un contorno ya parece sospechosamente grande (experimento 3), usar
el algoritmo de libro para separar objetos circulares que se tocan -- distance
transform + maximos locales como semillas -> watershed -- sobre **toda** la mascara de
foreground de golpe, para que sea la tecnica principal de deteccion y no un parche.

**Primer intento, fallido por un bug de diseno**: la primera version reutilizaba el
`adaptiveThreshold` de la variante C. Eso genero una mascara de foreground que cubria
30-40% de cada tile (cualquier zona algo mas clara que su entorno local: carreteras,
tejados, bordes de campo, no solo tanques), y ademas **nunca se marcaba explicitamente
el fondo** antes de llamar a `cv2.watershed` -- se dejaba en 0 ("desconocido") en vez de
asignarle una etiqueta de fondo seguro. Sin una semilla de fondo, `cv2.watershed` no
tiene desde donde crecer esa región y deja casi todo como frontera (`-1`) o sin asignar:
en la practica, el 99.97% de las ~3400 regiones resultantes por tile eran de 1 pixel
(`area media = 1.03px`, ver commit de este experimento). Es el error clasico de saltarse
el paso de "sure background / sure foreground / unknown" de la receta estandar de OpenCV
para watershed marker-controlled.

**Correccion**: (1) sustituir el umbral adaptativo local por un umbral global de
percentil de brillo (`brightness_percentile`, ej. top 3-5% mas claro de la escena), que
aisla mucho mejor los tejados de los tanques que cualquier variante de umbral local
probada; (2) construir explicitamente `sure_fg` (dilatacion pequeña de los maximos
locales), `sure_bg` (dilatacion grande de la mascara binaria) y `unknown = sure_bg -
sure_fg`, y pasar eso como marcadores a `cv2.watershed` -- la receta correcta.

**Resultado tras la correccion**: F1=0.247 en val, tercer mejor metodo clasico (por
detras de blob detector en sus dos versiones). Recall=0.217, mejor que Hough (ambas
variantes) y contornos, pero bastante mas ruido que el blob detector: sigue generando
muchos falsos positivos porque el umbral por percentil de brillo, aunque mas limpio que
el adaptativo, todavia captura muchas superficies claras que no son tanques
(edificios, hormigon, agua en calma reflectante).

## Experimento 7 — Deteccion multiescala Difference-of-Gaussians (`skimage.feature.blob_dog`)

`src/classic_methods.dog_blob_detect`. Hipotesis: Hough y `SimpleBlobDetector` operan a
una escala mas o menos fija por configuracion; un detector de blobs multiescala (DoG,
aproximacion rapida de Laplacian-of-Gaussian) barre un rango continuo de sigmas y podria
adaptarse mejor al rango real de radios del dataset (3-108px, con clusters de tanques de
tamanos muy distintos en la misma imagen).

Se empezo probando `blob_log` (la version "pura" antes de optimizar): **~3s por tile**,
inviable para grid search (68s por imagen completa). Se cambio a `blob_dog`
(aproximacion por piramide de diferencias de Gaussianas), 4-10x mas rapido, pero aun asi
el metodo mas lento de todos los probados (hasta 3.36s/imagen en el grid final).

**Resultado**: F1=0.039, el peor de todos los metodos deterministas probados, incluso
por debajo del Hough Circles ingenuo de imagen completa (0.063). Causa mas probable:
`blob_dog`, tal y como esta usado aqui, no distingue polaridad de blob (no hay
equivalente al `blob_color` de `SimpleBlobDetector`) -- responde igual a manchas claras
sobre fondo oscuro que a manchas oscuras sobre fondo claro, y el experimento 4 ya habia
demostrado que la señal util esta *solo* en los blobs claros (`blob_color=0` daba F1<0.01
en casi toda la grid). Sin ese filtro de polaridad, DoG dispara sobre practicamente
cualquier variacion de contraste local de la escena (sombras, bordes de parcelas,
carreteras), generando ~5400 falsos positivos en 20 imagenes. Una version corregida
tendria que combinarse con la misma mascara de brillo del experimento 6 antes de correr
DoG, pero dado que ya es el metodo mas lento y que la hipotesis de partida (adaptacion
multiescala) no compensa ese problema de fondo, no se siguio refinando.

## Experimento 8 — Post-proceso: filtro por densidad de vecinos

`src/classic_methods.filter_by_density`, aplicado sobre el mejor detector hasta ahora
(blob detector refinado, experimento 5). Hipotesis: los tanques reales casi nunca
aparecen aislados -- se instalan en "granjas de tanques" (mediana ~100 tanques/imagen) --
asi que una deteccion sin ninguna otra cerca es probablemente ruido (un tejado, una
piscina, un parche de terreno claro) y se puede descartar sin apenas coste de recall.

Se cachearon las cajas del blob detector una sola vez por imagen (la parte cara) y se
probaron 15 combinaciones de `(radius, min_neighbors)` sobre esas cajas ya calculadas
(el filtro en si es casi gratis), tanto en tuning como en la evaluacion final.

**Resultado**: F1=0.355 en val, practicamente identico al 0.354 sin filtro (mejor config:
`radius=300, min_neighbors=2`). Sube la precision (0.584 -> 0.626) pero baja el recall
casi lo mismo (0.254 -> 0.248), asi que el efecto neto en F1 es nulo. Revisando la
tendencia en la grid de tuning: con radios pequeños (100px) y min_neighbors alto (3) la
precision sube mucho mas (0.667) pero el recall cae proporcionalmente mas, y ninguna
combinacion probada rompe ese empate.

**Interpretacion**: la hipotesis de partida (los tanques reales estan en clusters) es
correcta, pero **los falsos positivos tambien lo estan** -- no son ruido disperso
aleatorio por toda la imagen sino, sobre todo, otras estructuras hechas por el hombre que
tambien vienen en grupos (naves industriales, urbanizaciones, filas de casas), que es
justo el tipo de escena donde aparecen las granjas de tanques reales. La densidad
espacial por si sola no separa "cluster de tanques" de "cluster de tejados similares";
haria falta una señal adicional (color/textura especifica del tanque, no solo su
disposicion geometrica) para que este filtro aportase algo. Se descarta como mejora neta,
pero queda documentado el resultado porque es un negativo instructivo: confirma que el
techo de F1~0.35 no se rompe con post-procesos baratos sobre las mismas cajas, hace falta
cambiar la señal de deteccion en si.

## Conclusion

El techo del metodo determinista en este dataset esta alrededor de **F1~0.35** (blob
detector tileado y afinado), muy por debajo del **F1=0.843** de YOLOv8 fine-tuned. La
causa raiz no es un problema de tuning insuficiente sino estructural: los detectores
clasicos de forma (Hough, contornos, blobs) razonan sobre un tanque a la vez a partir de
bordes/umbrales locales, y este dataset tiene hasta 893 tanques por imagen colocados en
rejillas muy juntas -- exactamente el escenario en el que separar instancias que se
tocan es mas dificil sin contexto de mas alto nivel (que es lo que aporta una CNN
entrenada con ejemplos reales de esos clusters).

Lo que si demuestra la serie de experimentos es que la eleccion de tecnica clasica
importa mucho mas que su tuning: pasar de Hough a blob detection (misma logica de
tiling, mismo protocolo de evaluacion) multiplico el F1 por 2 sin cambiar nada mas, y
tilear en vez de trabajar sobre la imagen completa fue la segunda mejora mas grande.
Ademas, el IoU medio de los aciertos (~0.56-0.81 en todos los metodos clasicos, similar
al de YOLO en los mejores casos) confirma que el cuello de botella de estos metodos es
sobre todo de **recall/deteccion**, no de precision geometrica de la caja.

Con 7 variantes deterministas probadas (Hough completo, Hough tileado, contornos +
circularidad + watershed local, blob detector x2, watershed universal, DoG multiescala),
el patron que emerge es claro: **cualquier tecnica que module explicitamente por
polaridad de brillo** (blob detector con `blob_color=255`, watershed sobre umbral de
percentil de brillo) supera con margen a las que solo miran bordes/forma sin ese filtro
(Hough, DoG sin restriccion de polaridad). Es la señal mas fuerte y mas barata de aplicar
en este dataset -- los tanques son, ante todo, los objetos mas claros y compactos de la
escena -- y ninguna de las variantes que la ignora pasa de F1~0.17.

El experimento 8 (filtro de densidad) confirma ademas que **el techo no se rompe con
post-procesado barato sobre las mismas cajas**: filtrar por contexto geometrico
(clusters) no discrimina tanques de otros clusters de estructuras humanas con el mismo
patron espacial. Para superar F1~0.35 haria falta una señal de deteccion distinta (color
real en vez de solo brillo en escala de grises, o directamente contexto aprendido de
ejemplos -- que es lo que hace la CNN).

## Archivos relevantes

- `src/classic_methods.py` — Hough tileado, blob detector, contornos+watershed local,
  watershed universal, DoG multiescala, filtro de densidad (7 detectores + 1 post-proceso).
- `src/tiling.py` — helper de tiling compartido entre YOLO y los metodos clasicos.
- `src/evaluate.py` — matching por IoU, metricas agregadas y NMS generico.
- `scripts/experiment_classic.py` — grid search + evaluacion de Hough tileado, blob
  detector (grid inicial) y contornos+circularidad.
- `scripts/refine_blob.py` — segunda pasada de tuning fino sobre blob detector.
- `scripts/experiment_classic2.py` — grid search + evaluacion de watershed universal y
  DoG multiescala.
- `scripts/experiment_density_filter.py` — grid search del post-proceso de densidad
  sobre las cajas ya calculadas del blob detector refinado.
- `results/hough_metrics.json`, `results/classic_experiments.json`,
  `results/blob_detector_refined.json`, `results/density_filter.json`,
  `results/yolo_metrics.json` — metricas crudas.
- `results/figures/blob_vs_yolo_*.jpg` — comparativa visual GT/blob/YOLO.
