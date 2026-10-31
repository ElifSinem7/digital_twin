#!/usr/bin/env python3
"""
build_static_mask.py
======================
TUM videodan piksel duzeyinde "sabit nesne" maskesi kurar: YOLO'nun arac sandigi ama hic
hareket etmeyen nesneler (sinyal kafalari, direkler, yoldaki ok isaretleri...).

Yontem
  1. Videonun tamami ~1 kare/s orneklenir, YOLO (arac siniflari, conf >= 0.25, sinif-bagimsiz NMS).
  2. Kutular SIKI kumelenir: ayni kumeye girmek icin IoU >= 0.85 (neredeyse ayni piksel kutusu).
  3. Bir kume SABIT NESNE sayilir eger
       - orneklenen karelerin >= %2'sinde (>= ~70 s) goruluyorsa,
       - gorulmeleri en az 10 dakikaya yayiliyorsa (tek bir uzun kirmizi isik degil),
       - gorunumu her seferinde ayni ise: kume icindeki 16x16 gri yamalarin medyan yamaya
         normalize capraz korelasyonunun medyani >= 0.8 (farkli arabalar ayni yerde dursa
         gorunumleri farkli olur -> maskelenmez).
  4. Maske = sabit kutular (+4 px) + istege bagli elle cizilmis poligonlar
     (mirror/static_mask_manual.json: {"polygons": [[[x,y],...], ...]}).

Cikti: mirror/static_objects.json, mirror/static_mask.png (0/255), demo/static_mask.png (kontrol gorseli)
Not: pandas gerektirmez (GPU venv'inde calisir).
"""
import argparse
import csv
import json
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VEHICLE_CLASSES = [2, 3, 5, 7]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-9)


def patch(gray, b):
    x1, y1, x2, y2 = [int(round(v)) for v in b]
    p = gray[max(y1, 0):max(y2, y1 + 1), max(x1, 0):max(x2, x1 + 1)]
    if p.size == 0:
        return None
    p = cv2.resize(p, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    p -= p.mean()
    n = np.linalg.norm(p)
    return p / n if n > 1e-6 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(ROOT, "videos", "Bellevue_116th_NE12th__2017-09-10_19-08-25.mp4"))
    ap.add_argument("--model", default=os.path.join(ROOT, "scripts", "yolov8m.pt"))
    ap.add_argument("--every-s", type=float, default=1.0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--min-frac", type=float, default=0.02)
    ap.add_argument("--min-spread-s", type=float, default=600)
    ap.add_argument("--min-ncc", type=float, default=0.8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--manual", default=os.path.join(HERE, "static_mask_manual.json"))
    ap.add_argument("--old-tracks", default=os.path.join(ROOT, "tracks", "2017-09-10_19.csv"))
    ap.add_argument("--check-time", type=float, default=445.0)
    ap.add_argument("--max-area", type=float, default=8000.0,
                    help="Bundan buyuk (px^2) kutular sabit nesne sayilmaz (yol yuzeyi olabilir, gercek araclari silerdi)")
    ap.add_argument("--reuse", action="store_true",
                    help="Videoyu yeniden taramadan mevcut static_objects.json'daki kumelerden maskeyi yeniden kur")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, int(round(args.every_s * fps)))
    clusters = []   # dict: box (running mean), hits, times, patches
    n_samp, fidx, t_wall = 0, 0, time.time()
    check_frame = None
    prev = None
    if args.reuse:
        prev = json.load(open(os.path.join(HERE, "static_objects.json")))
        n_samp = prev["n_sampled_frames"]
        for _ in range(int(round(args.check_time * fps))):
            cap.grab()
        check_frame = cap.read()[1]
    model = None if args.reuse else YOLO(args.model)
    while not args.reuse:
        ok = cap.grab()
        if not ok:
            break
        if fidx % stride == 0 or fidx == int(round(args.check_time * fps)):
            ok, frame = cap.retrieve()
            if not ok:
                break
            if fidx == int(round(args.check_time * fps)):
                check_frame = frame.copy()
            if fidx % stride == 0:
                n_samp += 1
                t = fidx / fps
                r = model.predict(frame, conf=args.conf, classes=VEHICLE_CLASSES, imgsz=1280, agnostic_nms=True,
                                  device=args.device, verbose=False)[0]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                for b in r.boxes.xyxy.cpu().numpy():
                    best, bi = 0.0, -1
                    for i, c in enumerate(clusters):
                        v = iou(b, c["box"])
                        if v > best:
                            best, bi = v, i
                    if best >= 0.85:
                        c = clusters[bi]
                        c["box"] = (c["box"] * c["hits"] + b) / (c["hits"] + 1)
                        c["hits"] += 1; c["t1"] = t
                    else:
                        c = {"box": b.astype(float), "hits": 1, "t0": t, "t1": t, "patches": []}
                        clusters.append(c)
                    if len(c["patches"]) < 300:
                        p = patch(gray, b)
                        if p is not None:
                            c["patches"].append(p)
                if n_samp % 300 == 0:
                    print(f"    {fidx}/{n_total} kare ({n_samp} ornek, {len(clusters)} kume, {time.time() - t_wall:.0f}s)", flush=True)
        fidx += 1
    cap.release()

    static = []
    for c in clusters:
        frac = c["hits"] / n_samp
        if frac < args.min_frac or (c["t1"] - c["t0"]) < args.min_spread_s:
            continue
        P = np.array(c["patches"])
        med = np.median(P, axis=0); med -= med.mean(); med /= max(np.linalg.norm(med), 1e-6)
        ncc = float(np.median(P @ med))
        static.append({"box": [round(float(v), 1) for v in c["box"]], "frac": round(frac, 4),
                       "seen_s": round(c["hits"] * args.every_s, 0), "t0": round(c["t0"], 1), "t1": round(c["t1"], 1),
                       "ncc": round(ncc, 3)})
    if prev is not None:
        static = prev["objects"]
    for s in static:
        x1, y1, x2, y2 = s["box"]
        s["area"] = round((x2 - x1) * (y2 - y1), 0)
        s["static"] = bool(s["ncc"] >= args.min_ncc and s["area"] <= args.max_area)
        s["reason"] = ("sabit" if s["static"] else
                       "gorunum degisiyor" if s["ncc"] < args.min_ncc else "cok buyuk (yol yuzeyi)")
    static.sort(key=lambda s: -s["frac"])
    H, W = check_frame.shape[:2]
    mask = np.zeros((H, W), np.uint8)
    for s in static:
        if s["static"]:
            x1, y1, x2, y2 = s["box"]
            cv2.rectangle(mask, (int(x1) - 4, int(y1) - 4), (int(x2) + 4, int(y2) + 4), 255, -1)
    manual = json.load(open(args.manual))["polygons"] if os.path.exists(args.manual) else []
    for poly in manual:
        cv2.fillPoly(mask, [np.int32(poly)], 255)
    cv2.imwrite(os.path.join(HERE, "static_mask.png"), mask)
    with open(os.path.join(HERE, "static_objects.json"), "w") as f:
        json.dump({"n_sampled_frames": n_samp, "every_s": args.every_s, "objects": static,
                   "manual_polygons": manual}, f, indent=1)

    # capraz kontrol: eski tum-saat izlerindeki sabit izler (merkez std < 5 px, > 30 s) maskede mi?
    old = {}
    if os.path.exists(args.old_tracks):
        with open(args.old_tracks) as f:
            for r in csv.DictReader(f):
                old.setdefault(r["track_id"], []).append((float(r["time_s"]), float(r["cx"]), float(r["cy"])))
    old_static = []
    for tid, v in old.items():
        a = np.array(v)
        if a[-1, 0] - a[0, 0] > 30 and np.hypot(a[:, 1].std(), a[:, 2].std()) < 5:
            old_static.append((tid, a[:, 1].mean(), a[:, 2].mean(), a[-1, 0] - a[0, 0]))
    inside = [mask[int(round(y)), int(round(x))] > 0 for _, x, y, _ in old_static]

    vis = check_frame.copy()
    over = vis.copy(); over[mask > 0] = (0, 0, 255)
    vis = cv2.addWeighted(over, 0.45, vis, 0.55, 0)
    for s in static:
        x1, y1, x2, y2 = [int(v) for v in s["box"]]
        col = (0, 0, 255) if s["static"] else (0, 200, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 1)
        cv2.putText(vis, f"{s['frac'] * 100:.0f}% ncc{s['ncc']:.2f}", (x1, max(y1 - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (255, 255, 255), 1, cv2.LINE_AA)
    for (tid, x, y, d), ins in zip(old_static, inside):
        cv2.drawMarker(vis, (int(x), int(y)), (0, 255, 0) if ins else (255, 0, 255), cv2.MARKER_CROSS, 10, 2)
    bar = np.full((48, W, 3), 25, np.uint8)
    n_st = sum(s["static"] for s in static)
    cv2.putText(bar, f"Kirmizi: sabit-nesne maskesi ({n_st} nesne; tum video {n_samp} karede orneklendi). Turuncu cerceve: "
                f"tekrarlayan ama maskelenmeyen kume (gorunumu degisiyor ya da > {args.max_area:.0f} px2 yol yuzeyi).",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, f"Arti: eski tum-saat izlerindeki sabit izler - yesil: maskede ({sum(inside)}/{len(inside)}), mor: maske disinda. "
                f"Kare t={args.check_time:.0f} s.", (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(ROOT, "demo", "static_mask.png"), np.vstack([vis, bar]))
    print(f">>> {n_samp} ornek kare, {len(clusters)} kume; tekrarlayan aday {len(static)}, sabit nesne {n_st}")
    for s in static:
        print(f"    {'SABIT ' if s['static'] else 'aday  '} kutu {s['box']} ({s['area']:.0f} px2) goruldu %{s['frac'] * 100:.1f} "
              f"({s['seen_s']:.0f} s, {s['t0']:.0f}-{s['t1']:.0f} s) ncc {s['ncc']} -> {s['reason']}")
    print(f">>> Eski sabit izlerin {sum(inside)}/{len(inside)} tanesi maskede")
    print(f">>> mirror/static_mask.png, mirror/static_objects.json, demo/static_mask.png")


if __name__ == "__main__":
    main()
