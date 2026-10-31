#!/usr/bin/env python3
"""
compute_homography.py
======================
Videodaki piksel koordinatlari ile net.xml'in GERCEK metre koordinat
sistemi (SUMO/UTM) arasinda bir HOMOGRAFI (perspektif donusum) matrisi
hesaplar.

** KRITIK KURAL: SADECE ZEMIN SEVIYESINDEKI (yerdeki) noktalari kullanin **
Yaya gecidi cizgisi kosesi, dur cizgisi ucu, serit kesisimi, bordur kosesi
gibi seyler UYGUNDUR. Bina kosesi, direk tepesi, agac gibi YERDEN YUKSEK
seyler KULLANMAYIN - kamera acili cektigi icin bunlar "paralaks hatasi"
denen buyuk bir sapmaya yol acar (uydu goruntusu tepeden, video acili
gorunce yuksek noktalar farkli yerlerde gorunur).

Kurulum:
    pip install opencv-python numpy pandas

Kullanim:
    python compute_homography.py --points ../points.csv --out ../homography.json
"""
import argparse
import json

import cv2
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True,
                     help="CSV: pixel_x,pixel_y,real_x,real_y (basliksiz)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.points, header=None,
                      names=["pixel_x", "pixel_y", "real_x", "real_y"])

    if len(df) < 4:
        raise SystemExit(
            f"HATA: en az 4 eslesme noktasi gerekli, {len(df)} tane var."
        )

    src = df[["pixel_x", "pixel_y"]].values.astype(np.float32)
    dst = df[["real_x", "real_y"]].values.astype(np.float32)

    H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)

    if H is None:
        raise SystemExit("HATA: homografi hesaplanamadi.")

    n_inliers = int(mask.sum())
    print(f">>> {len(df)} noktadan {n_inliers} tanesi tutarli (inlier) kabul edildi.\n")

    # HER NOKTANIN AYRI ayri hatasini goster - hangisinin sorunlu oldugunu bulmak icin
    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(projected - dst, axis=1)

    print(f"{'#':<3} {'piksel':<16} {'gercek':<20} {'tahmin':<20} {'hata(m)':<10} {'inlier?'}")
    for i in range(len(df)):
        is_inlier = "EVET" if mask[i] else "HAYIR (disarida)"
        print(f"{i+1:<3} ({df.iloc[i]['pixel_x']:.0f},{df.iloc[i]['pixel_y']:.0f})"
              f"{'':<4} ({df.iloc[i]['real_x']:.1f},{df.iloc[i]['real_y']:.1f})"
              f"{'':<4} ({projected[i][0]:.1f},{projected[i][1]:.1f})"
              f"{'':<4} {errors[i]:<10.2f} {is_inlier}")

    print(f"\n>>> Ortalama hata: {errors.mean():.2f} m, Maksimum: {errors.max():.2f} m")

    worst_idx = int(np.argmax(errors))
    print(f">>> EN KOTU nokta: #{worst_idx+1} (hata={errors[worst_idx]:.2f}m) - "
          f"bu noktayi kontrol edin, muhtemelen yanlis eslesme ya da yerden "
          f"yuksek bir nokta (bina/direk) secilmis.")

    if errors.mean() > 3.0:
        print(">>> UYARI: ortalama hata hala yuksek. En kotu 1-2 noktayi cikarip "
              "(points.csv'den o satiri silip) tekrar deneyin, ya da o noktalari "
              "zemin seviyesinde baska bir referansla degistirin.")

    with open(args.out, "w") as f:
        json.dump({"homography_matrix": H.tolist(),
                   "mean_error_m": float(errors.mean()),
                   "n_points": len(df),
                   "n_inliers": n_inliers}, f, indent=2)

    print(f"\n>>> Kaydedildi: {args.out}")


if __name__ == "__main__":
    main()