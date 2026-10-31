#!/usr/bin/env python3
"""
apply_homography.py'nin urettigi, GERCEK metre koordinatlarina (real_x,
real_y) donusturulmus tracks dosyasindan arac hizlarini hesaplar.
"""
import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True,
                     help="apply_homography.py çıktısı (real_x, real_y sütunları olmalı)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-points", type=int, default=3)
    args = ap.parse_args()

    df = pd.read_csv(args.tracks)
    if df.empty:
        raise SystemExit("HATA: tracks dosyası boş.")
    if "real_x" not in df.columns or "real_y" not in df.columns:
        raise SystemExit(
            "HATA: 'real_x'/'real_y' sütunları yok. Önce apply_homography.py ile bu dosyayı üretmelisiniz."
        )

    rows = []
    for track_id, g in df.sort_values("time_s").groupby("track_id"):
        if len(g) < args.min_points:
            continue
        g = g.reset_index(drop=True)

        dx = g["real_x"].diff()
        dy = g["real_y"].diff()
        dt = g["time_s"].diff()
        dist_m = np.hypot(dx, dy)
        speed_mps = (dist_m / dt).replace([np.inf, -np.inf], np.nan).dropna()

        if speed_mps.empty:
            continue

        # Medyan kullaniyoruz (ortalama degil) - YOLO/tracking'in ara sira verdigi ani sicramalara (outlier) karsi daha dayanikli
        avg_speed_mps = speed_mps.median()
        p90_speed_mps = speed_mps.quantile(0.9)

        rows.append({
            "track_id": track_id,
            "cls": g["cls"].mode().iloc[0] if not g["cls"].mode().empty else "car",
            "n_points": len(g),
            "avg_speed_mps": round(avg_speed_mps, 2),
            "avg_speed_kmh": round(avg_speed_mps * 3.6, 1),
            "p90_speed_mps": round(p90_speed_mps, 2),
            "p90_speed_kmh": round(p90_speed_mps * 3.6, 1),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)

    print(f">>> {len(out_df)} araç için hız hesaplandı -> {args.out}")
    if not out_df.empty:
        overall_mps = out_df["avg_speed_mps"].mean()
        print(f">>> Ortalama hız (tüm araçlar): {overall_mps * 3.6:.1f} km/s "
              f"({overall_mps:.1f} m/s)")
        print(f">>> Bu değeri routes.rou.xml'deki vType maxSpeed alanına yaz.")
        print(f">>> Tip bazında ortalamalar:")
        print(out_df.groupby("cls")["avg_speed_kmh"].mean().round(1).to_string())


if __name__ == "__main__":
    main()