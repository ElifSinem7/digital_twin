#!/usr/bin/env python3
"""
Sinyal renginin KENDISINE bakmak yerine, arac hareketinden sinyal
fazlarini cikarir - isik/golge/gunes yansimasi gibi sorunlardan
ETKILENMEZ, cunku renk degil HAREKET analiz ediyor.

Mantik: bir yon kirmizi oldugunda, o yondeki araclar kavsak onunde
birikip DURUR (hiz ~0). Yesil oldugunda hep birlikte HAREKETE GECERLER.
Bu "dur-kalk" gecisini, o yaklasim kolundaki araclarin ortalama anlik
hizini zamana gore izleyerek tespit ediyoruz.

Kurulum:
    pip install pandas numpy shapely

Kullanim:
    python infer_signal_from_traffic.py \
        --tracks ../tracks/2017-09-10_19_real.csv \
        --config ../config.json \
        --zone entry_north \
        --out ../signal_from_traffic_north.csv \
        --stopped-threshold 1.0 \
        --time-bucket 1.0
"""
import argparse
import json

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True,
                     help="apply_homography.py çıktısı (real_x/real_y sütunları olmalı)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--zone", required=True,
                     help="config.json'daki hangi bölgeyi (örnek: entry_north) analiz edeceğiz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stopped-threshold", type=float, default=1.0,
                     help="Bu hızın (m/s) altındaki araçlar 'duruyor' sayılır")
    ap.add_argument("--time-bucket", type=float, default=1.0,
                     help="Kaç saniyelik zaman dilimlerinde ortalama alınacak")
    args = ap.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    if args.zone not in config["zones"]:
        raise SystemExit(f"HATA: '{args.zone}' config.json'da bulunamadı. "
                          f"Mevcut bölgeler: {list(config['zones'].keys())}")
    zone_poly = Polygon(config["zones"][args.zone])

    df = pd.read_csv(args.tracks)
    if "real_x" not in df.columns:
        raise SystemExit("HATA: 'real_x'/'real_y' sütunları yok - önce apply_homography.py çalıştırın.")

    # Sadece bu bolgenin icinde olan noktalari filtrele
    inside_mask = df.apply(lambda r: zone_poly.contains(Point(r["cx"], r["cy"])), axis=1)
    zdf = df[inside_mask].copy()
    print(f">>> '{args.zone}' bölgesi içinde {len(zdf)} nokta (toplam {len(df)} noktadan)")

    if zdf.empty:
        raise SystemExit("HATA: bu bölgede hiç nokta yok - config.json'daki bölgeyi kontrol edin.")

    # Her aracin ardisik konumlari arasindaki ANLIK hizini hesapla
    zdf = zdf.sort_values(["track_id", "time_s"])
    speeds = []
    for track_id, g in zdf.groupby("track_id"):
        g = g.reset_index(drop=True)
        dx, dy, dt = g["real_x"].diff(), g["real_y"].diff(), g["time_s"].diff()
        spd = (np.hypot(dx, dy) / dt).replace([np.inf, -np.inf], np.nan)
        g["inst_speed_mps"] = spd
        speeds.append(g)
    zdf = pd.concat(speeds).dropna(subset=["inst_speed_mps"])

    # Zamani kovalara (bucket) bolup, her kovadaki ORTALAMA hizi hesapla
    zdf["time_bucket"] = (zdf["time_s"] // args.time_bucket) * args.time_bucket
    bucket_speed = zdf.groupby("time_bucket")["inst_speed_mps"].median().reset_index()
    bucket_speed["state"] = np.where(
        bucket_speed["inst_speed_mps"] < args.stopped_threshold, "DURUYOR", "HAREKETLİ"
    )

    # Durum degisikliklerini (faz gecislerini) bul
    phases = []
    cur_state, seg_start = None, None
    for _, row in bucket_speed.iterrows():
        if row["state"] != cur_state:
            if cur_state is not None:
                phases.append((cur_state, seg_start, row["time_bucket"]))
            cur_state = row["state"]
            seg_start = row["time_bucket"]
    if cur_state is not None:
        phases.append((cur_state, seg_start, bucket_speed["time_bucket"].iloc[-1]))

    phase_df = pd.DataFrame(phases, columns=["durum", "baslangic_s", "bitis_s"])
    phase_df["sure_s"] = (phase_df["bitis_s"] - phase_df["baslangic_s"]).round(1)
    phase_df.to_csv(args.out, index=False)

    print(f">>> {len(phase_df)} durum deĞiŞikliĞi tespit edildi -> {args.out}\n")
    for durum in ["DURUYOR", "HAREKETLİ"]:
        d = phase_df[phase_df["durum"] == durum]["sure_s"]
        plausible = d[(d >= 2) & (d <= 120)]
        if len(plausible) > 0:
            print(f"  {durum:10s}: medyan={plausible.median():.1f}s, "
                  f"ortalama={plausible.mean():.1f}s, adet={len(plausible)}")

    print("\n>>> Yorumlama: 'DURUYOR' süreleri o yönün KIRMIZI olduğu süreye,")
    print("    'HAREKETLİ' süreleri YEŞİL (+devam eden akış) süresine yakın bir tahmindir.")


if __name__ == "__main__":
    main()