"""
Bir kavsak videosunu YOLOv8 + ByteTrack ile isleyip her karedeki her
aracin merkez noktasini (centroid) bir CSV'ye yazar. Bu CSV daha sonra
movement_to_routes.py tarafindan hangi aracin nereden nereye gittigini
(giris/cikis bolgeleri ile kesisimine bakarak) belirlemek icin kullanilir.
"""
import argparse
import os

import cv2
import pandas as pd
from ultralytics import YOLO

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="yolov8m.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--sample-every", type=int, default=2)
    ap.add_argument("--tracker", default="bytetrack.yaml")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"HATA: video açılamadı: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    model = YOLO(args.model)

    rows = []
    frame_idx = 0

    print(f">>> İşleniyor: {args.video} (fps={fps:.1f}, her {args.sample_every} karede bir)")

    results = model.track(
        source=args.video,
        conf=args.conf,
        classes=list(VEHICLE_CLASSES.keys()),
        tracker=args.tracker,
        persist=True,
        stream=True,
        vid_stride=args.sample_every,
        verbose=False,
    )

    for r in results:
        time_s = frame_idx * args.sample_every / fps
        if r.boxes is not None and r.boxes.id is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), tid, cls in zip(xyxy, ids, clss):
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                rows.append({
                    "frame": frame_idx,
                    "time_s": round(time_s, 2),
                    "track_id": int(tid),
                    "cls": VEHICLE_CLASSES.get(int(cls), "unknown"),
                    "cx": round(float(cx), 1),
                    "cy": round(float(cy), 1),
                })
        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"    ... {frame_idx} kare işlendi ({time_s:.0f} sn)")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f">>> Kaydedildi: {args.out}  ({len(df)} satır, {df['track_id'].nunique() if len(df) else 0} benzersiz araç)")


if __name__ == "__main__":
    main()