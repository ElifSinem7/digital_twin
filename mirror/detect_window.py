#!/usr/bin/env python3
"""
detect_window.py
==================
scripts/vehicle_tracker.py'nin (degistirilmeden birakildi) TAM KUTU kaydeden,
sadece belirli bir zaman araligi icin calisan versiyonu.

  - Kareler bastan SIRAYLA okunur (cap.set bu videoda VFR yuzunden kayiyor);
    zaman = kare_no / fps, yani tracks/*.csv'deki time_s ile ayni eksen.
  - Her kare islenir (stride 1), YOLOv8, araba/motosiklet/otobus/kamyon, SINIF-BAGIMSIZ NMS
    (ayni araca hem 'car' hem 'truck' kutusu dusmesin).
  - --mask verilirse: tespitler TAKIPTEN ONCE sabit-nesne maskesiyle (build_static_mask.py)
    suzulur, sonra ByteTrack'e verilir. Boylece sinyal kafasi / yol oku gibi nesneler hangi
    izde olursa olsun hic iz olusturmaz.
    Atma kurali: kutunun >= %60'i maskede VE kutu alani, kestigi sabit nesne kutusunun
    <= 2.5 katiysa (maskenin ustunden gecen gercek bir arac daha buyuk / farkli bir kutu verir).
  - Cikti: frame,time_s,track_id,cls,conf,x1,y1,x2,y2,bx,by  (bx,by = kutunun alt-ortasi)

Kullanim (GPU icin CUDA destekli bir Python ortamiyla):
    python detect_window.py --start 384 --end 504 \
        --mask static_mask.png --out ../tracks/window_384_504_boxes.csv
"""
import argparse
import csv
import json
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
HERE = os.path.dirname(os.path.abspath(__file__))


def load_tracker_cfg(name):
    from ultralytics.utils import IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml
    try:
        from ultralytics.utils import YAML
        cfg = YAML.load(check_yaml(name))
    except ImportError:  # eski ultralytics
        from ultralytics.utils import yaml_load
        cfg = yaml_load(check_yaml(name))
    return IterableSimpleNamespace(**cfg)


class StaticMask:
    def __init__(self, mask_png, objects_json):
        m = cv2.imread(mask_png, cv2.IMREAD_GRAYSCALE)
        self.integral = cv2.integral((m > 0).astype(np.uint8))
        objs = json.load(open(objects_json))["objects"] if objects_json and os.path.exists(objects_json) else []
        self.boxes = np.array([o["box"] for o in objs if o["static"]], float).reshape(-1, 4)

    def keep(self, xyxy, frac_thr=0.6, iou_thr=0.3):
        """Atilacak tespit: kutusunun >= frac_thr'i maskede VE bir sabit nesne kutusuna benziyor (IoU >= iou_thr).
        Ok isaretinin ustunden gecen bir aracin kutusu cok daha buyuk oldugundan IoU dusuk kalir -> korunur.
        Elle cizilmis poligonlar (sabit kutusu yok): sadece maskede olma orani yeterli."""
        keep = np.ones(len(xyxy), bool)
        H, W = self.integral.shape[0] - 1, self.integral.shape[1] - 1
        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            a, b = int(np.clip(x1, 0, W)), int(np.clip(x2, 0, W))
            c, d = int(np.clip(y1, 0, H)), int(np.clip(y2, 0, H))
            area = max((b - a) * (d - c), 1)
            inside = self.integral[d, b] - self.integral[c, b] - self.integral[d, a] + self.integral[c, a]
            if inside / area < frac_thr:
                continue
            best, overl = 0.0, False
            for sx1, sy1, sx2, sy2 in self.boxes:
                ix = max(0.0, min(x2, sx2) - max(x1, sx1)); iy = max(0.0, min(y2, sy2) - max(y1, sy1))
                if ix * iy > 0:
                    overl = True
                    u = (x2 - x1) * (y2 - y1) + (sx2 - sx1) * (sy2 - sy1) - ix * iy
                    best = max(best, ix * iy / max(u, 1e-9))
            if best >= iou_thr or not overl:
                keep[i] = False
        return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(os.path.dirname(HERE), "videos",
                                                    "Bellevue_116th_NE12th__2017-09-10_19-08-25.mp4"))
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.path.join(HERE, "..", "scripts", "yolov8m.pt"))
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--mask", default=None, help="sabit-nesne maskesi (build_static_mask.py)")
    ap.add_argument("--mask-objects", default=os.path.join(HERE, "static_objects.json"))
    ap.add_argument("--device", default=None, help="0 (GPU) ya da cpu; bos = otomatik")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"HATA: video acilamadi: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    first, last = int(round(args.start * fps)), int(round(args.end * fps))
    for _ in range(first):
        if not cap.grab():
            raise SystemExit("HATA: video baslangictan once bitti")

    model = YOLO(args.model)
    smask = StaticMask(args.mask, args.mask_objects) if args.mask else None
    from ultralytics.trackers.byte_tracker import BYTETracker
    try:   # eski ultralytics: (args, frame_rate); yeni (8.4+): sadece args (track_buffer kare cinsinden)
        tracker = BYTETracker(load_tracker_cfg(args.tracker), frame_rate=int(round(fps)))
    except TypeError:
        tracker = BYTETracker(load_tracker_cfg(args.tracker))
    rows, t_wall, n_drop, n_det = [], time.time(), 0, 0
    for fidx in range(first, last):
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, conf=args.conf, imgsz=args.imgsz, classes=list(VEHICLE_CLASSES),
                          agnostic_nms=True, device=args.device, verbose=False)[0]
        det = r.boxes.cpu().numpy()
        n_det += len(det)
        if smask is not None and len(det):
            keep = smask.keep(det.xyxy)
            n_drop += int((~keep).sum())
            det = det[keep]
        tracks = tracker.update(det, frame)
        for x1, y1, x2, y2, tid, score, cls, _ in tracks:
            rows.append({"frame": fidx, "time_s": round(fidx / fps, 3), "track_id": int(tid),
                         "cls": VEHICLE_CLASSES.get(int(cls), "car"), "conf": round(float(score), 3),
                         "x1": round(float(x1), 1), "y1": round(float(y1), 1),
                         "x2": round(float(x2), 1), "y2": round(float(y2), 1),
                         "bx": round(float(x1 + x2) / 2, 1), "by": round(float(y2), 1)})
        if (fidx - first) % 300 == 0:
            print(f"    kare {fidx - first}/{last - first}  t={fidx / fps:.1f}s  "
                  f"({time.time() - t_wall:.0f}s gecti)", flush=True)
    cap.release()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fields = ["frame", "time_s", "track_id", "cls", "conf", "x1", "y1", "x2", "y2", "bx", "by"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    n_ids = len({r["track_id"] for r in rows})
    print(f">>> {n_det} tespit, maskeyle atilan {n_drop} ({n_drop / max(n_det, 1):.1%}); "
          f"{len(rows)} izli kutu, {n_ids} iz -> {args.out}  ({time.time() - t_wall:.0f}s)")


if __name__ == "__main__":
    main()
