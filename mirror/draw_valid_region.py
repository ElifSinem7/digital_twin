#!/usr/bin/env python3
"""Ayna modu gecerli bolgesini (kavsak merkezine <= 40 m, kameranin gordugu zemin) video karesine cizer
ve 384-504 s tespitlerinin alt-orta noktalarini icinde/disinda olarak isaretler -> demo/mapping_valid_region.png"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Point

from fisheye_map import FisheyeGroundMap
from fit_mapping import grab_frame
from view_utils import JUNCTION_C, polygon_to_image, visible_polygon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-dist", type=float, default=40.0)
    ap.add_argument("--time", type=float, default=421.0)
    args = ap.parse_args()
    m = FisheyeGroundMap.load(os.path.join(HERE, "fisheye_mapping.json"))
    region = visible_polygon(m, args.max_dist)
    fr = grab_frame(os.path.join(ROOT, "videos", "Bellevue_116th_NE12th__2017-09-10_19-08-25.mp4"), args.time)
    img = fr.copy()
    bx = pd.read_csv(os.path.join(ROOT, "tracks", "window_384_504_boxes.csv"))
    w = m.img_to_world(bx[["bx", "by"]].to_numpy())
    d = np.hypot(w[:, 0] - JUNCTION_C[0], w[:, 1] - JUNCTION_C[1])
    inside = np.isfinite(d) & (d <= args.max_dist)
    for (x, y), ok in zip(bx[["bx", "by"]].to_numpy()[::3], inside[::3]):
        cv2.circle(img, (int(x), int(y)), 2, (0, 220, 0) if ok else (0, 0, 255), -1)
    p = polygon_to_image(m, region)
    p = p[np.isfinite(p).all(1)]
    cv2.polylines(img, [np.int32(p)], True, (255, 255, 0), 2, cv2.LINE_AA)
    # 20/30 m halkalari referans icin
    for r in (20, 30):
        ring = Point(JUNCTION_C).buffer(r, 128)
        q = polygon_to_image(m, ring)
        q = q[np.isfinite(q).all(1)]
        for a, b in zip(q[:-1], q[1:]):
            if np.hypot(*(a - b)) < 30:
                cv2.line(img, tuple(np.int32(a)), tuple(np.int32(b)), (200, 200, 200), 1, cv2.LINE_AA)
    bar = np.full((48, img.shape[1], 3), 25, np.uint8)
    t1 = (f"Camgobegi: ayna modu gecerli bolgesi (kavsak merkezine <= {args.max_dist:.0f} m ve kameranin gordugu zemin). "
          f"Gri: 20 / 30 m halkalari.")
    t2 = (f"Yesil: bolge icindeki tespitler, kirmizi: atilanlar (384-504 s: {int((~inside).sum())}/{len(inside)} "
          f"kutu = {(~inside).mean():.1%} disarida)")
    cv2.putText(bar, t1, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, t2, (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1, cv2.LINE_AA)
    out = os.path.join(ROOT, "demo", "mapping_valid_region.png")
    cv2.imwrite(out, np.vstack([img, bar]))
    print(f">>> {out}  (bolge alani {region.area:.0f} m2; atilan kutu {(~inside).sum()}/{len(inside)})")


if __name__ == "__main__":
    main()
