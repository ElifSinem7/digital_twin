#!/usr/bin/env python3
"""
config.json'da isaretlediginiz sinyal kafalarinin (signal_heads) her
karedeki rengini (kirmizi/sari/yesil/sonuk) HSV renk esikleme ile okur.
Tum kafalarin renk vektoru degistigi anlari "faz gecisi" olarak isaretler
- boylece gercek sinyal dongusunu ELLE OLCMEDEN, videodan otomatik cikarir.

Kullanim:
    python signal_light_extractor.py \
        --video ../videos/Bellevue_116th_NE12th__2017-09-10_19-08-25.mp4 \
        --config ../config.json \
        --out-timeline ../signal_timeline_19.csv \
        --out-tls ../sumo/tls_auto.add.xml \
        --tls-id TLSID \
        --sample-fps 2
"""
import argparse
import json

import cv2
import numpy as np
import pandas as pd

# HSV renk araliklari (OpenCV: H 0-179, S 0-255, V 0-255)
COLOR_RANGES = {
    "red": [((0, 100, 100), (10, 255, 255)), ((170, 100, 100), (179, 255, 255))],
    "yellow": [((15, 100, 100), (35, 255, 255))],
    "green": [((40, 60, 60), (90, 255, 255))],
}


def classify_color(hsv_patch):
    best_color, best_score = "off", 0
    total_px = hsv_patch.shape[0] * hsv_patch.shape[1]
    if total_px == 0:
        return "off"
    for color, ranges in COLOR_RANGES.items():
        score = 0
        for lo, hi in ranges:
            mask = cv2.inRange(hsv_patch, np.array(lo), np.array(hi))
            score += cv2.countNonZero(mask)
        frac = score / total_px
        if frac > best_score:
            best_score = frac
            best_color = color
    # yeterince parlak piksel yoksa (lamba sonuk / kapali) "off" say
    return best_color if best_score > 0.05 else "off"


def state_char(color):
    return {"green": "G", "yellow": "y", "red": "r", "off": "r"}.get(color, "r")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-timeline", required=True)
    ap.add_argument("--out-tls", required=True)
    ap.add_argument("--tls-id", default="TLSID")
    ap.add_argument("--sample-fps", type=float, default=2.0)
    args = ap.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    signal_heads = config["signal_heads"]
    if not signal_heads:
        raise SystemExit("HATA: config.json içinde signal_heads tanımlı değil. Önce roi_selector.py ile işaretleyin.")
    head_names = sorted(signal_heads.keys())

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"HATA: video açılamadı: {args.video}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_stride = max(1, round(video_fps / args.sample_fps))

    rows = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_stride == 0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            time_s = frame_idx / video_fps
            row = {"frame": frame_idx, "time_s": round(time_s, 2)}
            for name in head_names:
                x, y, w, h = signal_heads[name]
                patch = hsv[y:y + h, x:x + w]
                color = classify_color(patch)
                # ONEMLI: dogrudan TEK HARFLIK SUMO koduna ceviriyoruz state_key'in her karakteri tam olarak bir sinyal kafasina karsilik gelsin diye.
                row[name] = state_char(color)
            rows.append(row)
        frame_idx += 1
    cap.release()

    df = pd.DataFrame(rows)
    df.to_csv(args.out_timeline, index=False)
    print(f">>> Zaman çizelgesi kaydedildi: {args.out_timeline} ({len(df)} ornek)")

    # Ayni renk vektorunun surdugu araliklari (fazlari) bul
    df["state_key"] = df[head_names].agg("".join, axis=1)
    phases = []
    cur_key = None
    phase_start = None
    for _, r in df.iterrows():
        if r["state_key"] != cur_key:
            if cur_key is not None:
                phases.append((cur_key, phase_start, r["time_s"]))
            cur_key = r["state_key"]
            phase_start = r["time_s"]
    if cur_key is not None:
        phases.append((cur_key, phase_start, df["time_s"].iloc[-1]))

    print(f">>> {len(phases)} faz geçişi tespit edildi (video süresince).")

    # tlLogic XML uret
    lines = ['<?xml version="1.0" encoding="UTF-8"?>']
    lines.append("<!-- OTOMATİK ÜRETİLDİ - signal_light_extractor.py -->")
    lines.append("<!-- state harflerinin sırasını netedit'teki gerçek connection")
    lines.append("     sırasıyla doğrulayın, sadece 'duration' değerleri doğrudan")
    lines.append("     videodan ölçülmüştür. -->")
    lines.append("<additional>")
    lines.append(f'    <tlLogic id="{args.tls_id}" type="static" '
                  f'programID="video-calibrated" offset="0">')
    for state_key, start, end in phases:
        duration = round(end - start, 1)
        if duration < 0.5:
            continue  # gurultu / gecici okuma hatasi, atla
        state_str = state_key
        lines.append(f'        <phase duration="{duration}" state="{state_str}"/>')
    lines.append("    </tlLogic>")
    lines.append("</additional>")

    with open(args.out_tls, "w") as f:
        f.write("\n".join(lines))
    print(f">>> Otomatik tls dosyası yazıldı: {args.out_tls}")
    print(">>> UYARI: state harflerinin sırasını netedit'te doğrulamayı unutma.")


if __name__ == "__main__":
    main()