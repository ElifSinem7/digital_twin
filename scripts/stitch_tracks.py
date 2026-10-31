#!/usr/bin/env python3
"""
stitch_tracks.py
==================
vehicle_tracker.py'nin urettigi tracks.csv icinde, YOLO/ByteTrack'in
GERCEKTE AYNI araci kaybedip yeniden tespit ederken FARKLI bir track_id
verdigi durumlari tespit edip BIRLESTIRIR (track stitching).

Mantik: bir izin (A) BITTIGI andaki konum ve hizindan, kisa bir sure
sonra (max-gap-s icinde) NEREDE olmasi beklenecegini hesaplariz
(extrapolasyon). Baska bir izin (B) BASLADIGI konum bu tahmine yeterince
yakinsa (max-dist-px icinde) ve arac tipi (cls) uyumluysa, A ve B'nin
AYNI gercek aracin iki parcasi oldugunu varsayip B'nin ID'sini A'nin
ID'sine cevirerek birlestiririz.

Bu, ozellikle giris/cikis bolgesi eslesme oranini artirmak icin onemli:
kopuk bir iz genelde ne girisi ne cikisi tek basina yakalayabiliyordu,
birlesince tek, butun bir iz haline geliyor.

Kurulum:
    pip install pandas numpy

Kullanim:
    python stitch_tracks.py \
        --tracks ../tracks/2017-09-10_19.csv \
        --out ../tracks/2017-09-10_19_stitched.csv \
        --max-gap-s 3.0 \
        --max-dist-px 80
"""
import argparse

import numpy as np
import pandas as pd


def estimate_velocity(rows, n_points=3):
    """Bir izin SON n_points noktasindan piksel/saniye hizini tahmin eder."""
    tail = rows.tail(n_points)
    if len(tail) < 2:
        return 0.0, 0.0
    dt = tail["time_s"].iloc[-1] - tail["time_s"].iloc[0]
    if dt <= 0:
        return 0.0, 0.0
    vx = (tail["cx"].iloc[-1] - tail["cx"].iloc[0]) / dt
    vy = (tail["cy"].iloc[-1] - tail["cy"].iloc[0]) / dt
    return vx, vy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-gap-s", type=float, default=3.0,
                     help="Iki iz arasinda en fazla kac saniye bosluk olabilir")
    ap.add_argument("--max-dist-px", type=float, default=80.0,
                     help="Tahmin edilen konumla gercek baslangic arasinda "
                          "en fazla kac piksel fark olabilir")
    args = ap.parse_args()

    df = pd.read_csv(args.tracks)
    if df.empty:
        raise SystemExit("HATA: tracks dosyasi bos.")

    df = df.sort_values(["track_id", "time_s"])
    tracks = {}
    for tid, rows in df.groupby("track_id"):
        rows = rows.reset_index(drop=True)
        vx, vy = estimate_velocity(rows)
        tracks[tid] = {
            "start_time": rows["time_s"].iloc[0],
            "start_pos": (rows["cx"].iloc[0], rows["cy"].iloc[0]),
            "end_time": rows["time_s"].iloc[-1],
            "end_pos": (rows["cx"].iloc[-1], rows["cy"].iloc[-1]),
            "vx": vx, "vy": vy,
            "cls": rows["cls"].mode().iloc[0] if not rows["cls"].mode().empty else "car",
        }

    tids = list(tracks.keys())
    print(f">>> Toplam {len(tids)} ham iz bulundu.")

    # Her (A_bitis, B_baslangic) cifti icin aday eslesmeleri bul
    candidates = []
    for a in tids:
        ta = tracks[a]
        for b in tids:
            if a == b:
                continue
            tb = tracks[b]
            gap = tb["start_time"] - ta["end_time"]
            if gap <= 0 or gap > args.max_gap_s:
                continue
            if ta["cls"] != tb["cls"]:
                continue
            pred_x = ta["end_pos"][0] + ta["vx"] * gap
            pred_y = ta["end_pos"][1] + ta["vy"] * gap
            dist = np.hypot(pred_x - tb["start_pos"][0], pred_y - tb["start_pos"][1])
            if dist <= args.max_dist_px:
                # skor: kucuk mesafe VE kucuk bosluk daha iyi
                score = dist + gap * 10
                candidates.append((score, a, b))

    candidates.sort(key=lambda c: c[0])

    next_of = {}   # a -> b (a'nin devami b)
    used_as_next = set()   # b'ler (birden fazla A'nin devami olamaz)
    used_as_prev = set()   # a'lar (bir A birden fazla B'ye devam edemez)

    for score, a, b in candidates:
        if a in used_as_prev or b in used_as_next:
            continue
        next_of[a] = b
        used_as_prev.add(a)
        used_as_next.add(b)

    # Zincirleri (A->B->C...) coz, her zincire TEK bir kanonik ID ata
    canonical = {}

    def find_root(t):
        # t'den geriye giderek zincirin BASINI bul
        preds = {v: k for k, v in next_of.items()}
        while t in preds:
            t = preds[t]
        return t

    for tid in tids:
        canonical[tid] = find_root(tid)

    n_merged = sum(1 for tid in tids if canonical[tid] != tid)
    n_final = len(set(canonical.values()))
    print(f">>> {len(next_of)} birlestirme yapildi.")
    print(f">>> {len(tids)} ham iz -> {n_final} birlestirilmis iz "
          f"({n_merged} iz baska bir ize dahil edildi).")

    df["track_id"] = df["track_id"].map(canonical)
    df = df.sort_values(["track_id", "time_s"])
    df.to_csv(args.out, index=False)
    print(f">>> Kaydedildi: {args.out}")


if __name__ == "__main__":
    main()