"""Helper compartido para trocear una imagen grande en tiles con overlap."""


def tile_origins(image_size, tile_size, stride):
    """Coordenadas de inicio (x o y) de los tiles a lo largo de una dimension."""
    origins = list(range(0, image_size - tile_size, stride)) + [image_size - tile_size]
    return sorted(set(o for o in origins if o >= 0))


def iter_tiles(image, tile_size, stride):
    """Genera (tx, ty, tile_array) recorriendo la imagen completa con overlap."""
    h, w = image.shape[:2]
    xs = tile_origins(w, tile_size, stride)
    ys = tile_origins(h, tile_size, stride)
    for ty in ys:
        for tx in xs:
            yield tx, ty, image[ty:ty + tile_size, tx:tx + tile_size]
