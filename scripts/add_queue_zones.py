#!/usr/bin/env python3
"""
Mevcut config.json'u BOZMADAN, dur cizgisine yakin yeni "kuyruk"
bolgeleri eklemenizi saglar (infer_signal_from_traffic.py icin - entry
bolgeleri kavsaga cok uzak oldugu icin araclar orada hep hareketli
gorunuyor, kuyruk analizinde ise yaramiyor).

Kullanim:
    python add_queue_zones.py --frame ../reference_frame.png --config ../config.json

Kontroller roi_selector.py ile ayni: sol tik nokta ekler, 'n' bitirir/isim
sorar, 'z' geri alir, 's' KAYDEDER (mevcut zone'lari SILMEZ, ustune ekler),
'q' cikar.

Bu sefer dur cizgisinin HEMEN ARKASINDAKI (araclarin kirmizida durup
bekledigi) dar bir seridi isaretleyin. Onerilen isimler:
    queue_north, queue_south, queue_east, queue_west
"""
import argparse
import json
import os

import cv2

current_points = []
zones = {}


def redraw(frame):
    disp = frame.copy()
    for name, pts in zones.items():
        pts_np = [tuple(p) for p in pts]
        for i in range(len(pts_np)):
            cv2.line(disp, pts_np[i], pts_np[(i + 1) % len(pts_np)], (0, 200, 255), 2)
        if pts_np:
            cv2.putText(disp, name, pts_np[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    for p in current_points:
        cv2.circle(disp, tuple(p), 4, (255, 0, 255), -1)
    if len(current_points) > 1:
        for i in range(len(current_points) - 1):
            cv2.line(disp, tuple(current_points[i]), tuple(current_points[i + 1]), (255, 0, 255), 1)
    cv2.imshow("add_queue_zones", disp)


def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append([x, y])
        redraw(param)


def main():
    global current_points
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    frame = cv2.imread(args.frame)
    if frame is None:
        raise SystemExit(f"HATA: kare okunamadı: {args.frame}")

    # MEVCUT config.json'u yukle - uzerine yazacagiz ama zones'u KORUYARAK
    if os.path.exists(args.config):
        with open(args.config) as f:
            existing = json.load(f)
        zones.update(existing.get("zones", {}))
        signal_heads = existing.get("signal_heads", {})
        print(f">>> Mevcut {len(zones)} bölge yüklendi, üzerine ekleyeceksiniz.")
    else:
        signal_heads = {}

    cv2.namedWindow("add_queue_zones")
    cv2.setMouseCallback("add_queue_zones", mouse_cb, param=frame)
    redraw(frame)
    print(__doc__)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("z") and current_points:
            current_points.pop()
            redraw(frame)
        elif key == ord("n"):
            if len(current_points) >= 3:
                name = input("Yeni bölge adı (örnek: queue_north): ").strip()
                if name:
                    zones[name] = current_points.copy()
            current_points = []
            redraw(frame)
        elif key == ord("s"):
            config = {
                "reference_frame": os.path.basename(args.frame),
                "zones": zones,
                "signal_heads": signal_heads,
            }
            with open(args.config, "w") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f">>> Kaydedildi: {args.config} (toplam {len(zones)} bölge)")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()