#!/usr/bin/env python3
"""
redraw_zone.py
================
Mevcut config.json'daki BELIRLI bir bolgeyi (ornek: entry_west) silip
yeniden cizmenizi saglar - diger tum bolgeler/sinyal kutulari KORUNUR.

Kullanim:
    python redraw_zone.py --frame ../reference_frame.png --config ../config.json --zone entry_west

Kontroller: roi_selector.py ile ayni (sol tik nokta, 'n' bitir/isim sor,
'z' geri al, 's' kaydet, 'q' cik).
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
        color = (0, 255, 0) if name.startswith("entry") else (0, 165, 255)
        pts_np = [tuple(p) for p in pts]
        for i in range(len(pts_np)):
            cv2.line(disp, pts_np[i], pts_np[(i + 1) % len(pts_np)], color, 2)
        if pts_np:
            cv2.putText(disp, name, pts_np[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    for p in current_points:
        cv2.circle(disp, tuple(p), 4, (255, 0, 255), -1)
    if len(current_points) > 1:
        for i in range(len(current_points) - 1):
            cv2.line(disp, tuple(current_points[i]), tuple(current_points[i + 1]), (255, 0, 255), 1)
    cv2.imshow("redraw_zone", disp)


def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append([x, y])
        redraw(param)


def main():
    global current_points
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--zone", required=True, help="Yeniden çizilecek bölgenin adı")
    args = ap.parse_args()

    frame = cv2.imread(args.frame)
    if frame is None:
        raise SystemExit(f"HATA: kare okunamadı: {args.frame}")

    with open(args.config) as f:
        existing = json.load(f)
    zones.update(existing.get("zones", {}))
    signal_heads = existing.get("signal_heads", {})

    if args.zone in zones:
        del zones[args.zone]
        print(f">>> '{args.zone}' silindi, şimdi yeniden çizin.")
    else:
        print(f">>> '{args.zone}' zaten yoktu, yeni olarak eklenecek.")

    cv2.namedWindow("redraw_zone")
    cv2.setMouseCallback("redraw_zone", mouse_cb, param=frame)
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
                name = input(f"Bölge adı (varsayılan: {args.zone}): ").strip() or args.zone
                zones[name] = current_points.copy()
                print(f">>> '{name}' kaydedildi (henüz diske yazılmadı, 's' ile kaydedin).")
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