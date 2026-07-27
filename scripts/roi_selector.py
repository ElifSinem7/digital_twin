"""
116th Ave NE & NE 12th St kavsaginin referans karesi (screenshot) uzerinde
tikliyarak:
  1) Her yaklasim kolu icin GIRIS bolgesi (arac ilk goruldugunde hangi
     yoldan geldigini anlamak icin)
  2) Her yaklasim kolu icin CIKIS bolgesi (arac kavsaktan hangi yola
     ciktigini anlamak icin)
  3) Her sinyal kafasinin (goruntude 3 tane oldugunu belirttiniz) kucuk
     bir kutusunu
isaretlemenizi saglar, sonucu config.json'a yazar.
Kontroller:
    - Sol tik: nokta ekle (bolge/poligon icin)
    - 'n'    : mevcut poligonu bitir, isim sor, yeni poligona basla
    - 'b'    : (box modunda) sinyal kafasi icin dikdortgen kutu cizmeye gec
    - 'z'    : son noktayi geri al
    - 's'    : config.json'a kaydet
    - 'q'    : cikis (kaydetmeden)

"""
import argparse
import json
import os

import cv2

current_points = []
zones = {}          # name -> list of [x, y] (polygon)
signal_boxes = {}    # name -> [x, y, w, h]
mode = "polygon"     # "polygon" veya "box"
box_start = None


def redraw(img, frame):
    disp = frame.copy()
    for name, pts in zones.items():
        color = (0, 255, 0) if name.startswith("entry") else (0, 165, 255)
        pts_np = [tuple(p) for p in pts]
        for i in range(len(pts_np)):
            cv2.line(disp, pts_np[i], pts_np[(i + 1) % len(pts_np)], color, 2)
        if pts_np:
            cv2.putText(disp, name, pts_np[0], cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

    for name, (x, y, w, h) in signal_boxes.items():
        cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(disp, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2)

    for p in current_points:
        cv2.circle(disp, tuple(p), 4, (255, 0, 255), -1)
    if len(current_points) > 1:
        for i in range(len(current_points) - 1):
            cv2.line(disp, tuple(current_points[i]), tuple(current_points[i + 1]),
                      (255, 0, 255), 1)
    cv2.imshow("ROI Selector", disp)


def mouse_cb(event, x, y, flags, param):
    global current_points, box_start
    frame = param
    if mode == "polygon" and event == cv2.EVENT_LBUTTONDOWN:
        current_points.append([x, y])
        redraw(frame, frame)
    elif mode == "box":
        if event == cv2.EVENT_LBUTTONDOWN:
            box_start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and box_start is not None:
            x0, y0 = box_start
            bx, by = min(x0, x), min(y0, y)
            bw, bh = abs(x - x0), abs(y - y0)
            name = input("Sinyal kafası adı (ornek: signal_A): ").strip()
            if name:
                signal_boxes[name] = [bx, by, bw, bh]
            box_start = None
            redraw(frame, frame)


def main():
    global mode, current_points

    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, help="Referans kare (png/jpg)")
    ap.add_argument("--out", default="../config.json")
    args = ap.parse_args()

    frame = cv2.imread(args.frame)
    if frame is None:
        raise SystemExit(f"HATA: kare okunamadı: {args.frame}")

    cv2.namedWindow("ROI Selector")
    cv2.setMouseCallback("ROI Selector", mouse_cb, param=frame)
    redraw(frame, frame)

    print(__doc__)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("z") and current_points:
            current_points.pop()
            redraw(frame, frame)
        elif key == ord("b"):
            mode = "box"
            current_points = []
            print(">>> Kutu (sinyal kafası) modu - sol tıkla sürükleyip bırakın")
        elif key == ord("n"):
            if mode == "polygon" and len(current_points) >= 3:
                name = input("Bölge adı (ornek: entry_A / exit_A): ").strip()
                if name:
                    zones[name] = current_points.copy()
            current_points = []
            mode = "polygon"
            redraw(frame, frame)
        elif key == ord("s"):
            config = {
                "reference_frame": os.path.basename(args.frame),
                "zones": zones,
                "signal_heads": signal_boxes,
            }
            with open(args.out, "w") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f">>> Kaydedildi: {args.out}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()