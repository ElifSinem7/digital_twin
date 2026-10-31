#!/usr/bin/env python3
"""
vehicle_tracker.py'nin urettigi tracks.csv icindeki (cx,cy) PIKSEL
konumlarini, compute_homography.py'nin hesapladigi donusum matrisiyle
GERCEK metre koordinatlarina (net.xml ile AYNI koordinat sistemi:
real_x, real_y) cevirir. Sonuc, orijinal tracks.csv'ye iki yeni sutun
eklenmis hali.
"""
import argparse
import json

import cv2
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--homography", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.homography) as f:
        h_data = json.load(f)
    H = np.array(h_data["homography_matrix"], dtype=np.float64)

    df = pd.read_csv(args.tracks)
    if df.empty:
        raise SystemExit("HATA: tracks dosyası boş.")

    pixels = df[["cx", "cy"]].values.astype(np.float32).reshape(-1, 1, 2)
    real = cv2.perspectiveTransform(pixels, H).reshape(-1, 2)

    df["real_x"] = real[:, 0].round(2)
    df["real_y"] = real[:, 1].round(2)

    df.to_csv(args.out, index=False)
    print(f">>> {len(df)} satır dönüştürüldü -> {args.out}")
    print(f">>> Örnek: piksel ({df['cx'].iloc[0]:.0f},{df['cy'].iloc[0]:.0f}) "
          f"-> gerçek ({df['real_x'].iloc[0]:.1f}, {df['real_y'].iloc[0]:.1f}) metre")


if __name__ == "__main__":
    main()