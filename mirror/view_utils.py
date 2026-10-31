#!/usr/bin/env python3
"""Kameranin SUMO'daki gorus alani / bakis yonu yardimcilari (shapely + numpy)."""
import math

import numpy as np
from shapely.geometry import Point, Polygon

JUNCTION_C = (595.36, 810.97)


def camera_bearing(m):
    """Goruntunun alt-ortasindan merkezine giden zemin dogrultusu: pusula acisi (0=kuzey, saat yonu)."""
    a, b = m.img_to_world([[640, 710], [640, 330]])
    return math.degrees(math.atan2(b[0] - a[0], b[1] - a[1])) % 360


def visible_polygon(m, max_dist=40.0, centre=JUNCTION_C):
    """Kameranin gordugu zemin (goruntu kenarlarinin izdusumu) ∩ merkeze <= max_dist daire."""
    W, H = m.img_size
    n = 200
    border = np.vstack([np.c_[np.linspace(0, W - 1, n), np.zeros(n)],
                        np.c_[np.full(n, W - 1), np.linspace(0, H - 1, n)],
                        np.c_[np.linspace(W - 1, 0, n), np.full(n, H - 1)],
                        np.c_[np.zeros(n), np.linspace(H - 1, 0, n)]])
    w = m.img_to_world(border)
    # ufkun otesindeki (gecersiz/cok uzak) kenar noktalarini daireye kirp: cok uzaga cek
    cx, cy = centre
    out = []
    for (x, y), (u, v) in zip(w, border):
        if not np.isfinite(x) or math.hypot(x - cx, y - cy) > 4 * max_dist:
            # goruntu ustu ufka yakin: o dogrultuda uzak bir nokta koy
            q = m.img_to_world([[u, min(v + 5, H - 1)]])[0]
            k = 1
            while (not np.isfinite(q).all()) and k < 100:
                q = m.img_to_world([[u, min(v + 5 * k, H - 1)]])[0]; k += 1
            if not np.isfinite(q).all():
                continue
            d = np.array(q) - centre
            q = np.array(centre) + d / max(np.linalg.norm(d), 1e-6) * 4 * max_dist
            out.append(q)
        else:
            out.append((x, y))
    poly = Polygon(out).buffer(0)
    return poly.intersection(Point(centre).buffer(max_dist, 128))


def polygon_to_image(m, poly, step=0.5):
    """Dunya poligonunun sinirini goruntuye izdusur (piksel listesi; goruntu disi NaN atilir)."""
    xs, ys = poly.exterior.xy
    pts = []
    for (x0, y0), (x1, y1) in zip(zip(xs[:-1], ys[:-1]), zip(xs[1:], ys[1:])):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
        for k in range(n):
            pts.append((x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n))
    p = m.world_to_img(np.array(pts))
    return p
