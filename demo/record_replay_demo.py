#!/usr/bin/env python3
"""
record_replay_demo.py
=======================
record_demo.py'den FARKLI olarak araclari SUMO'nun fizik modeline birakmaz:
videodaki her gercek aracin, homografi ile SUMO koordinatina cevrilmis
konumunu (tracks/*_real.csv: real_x, real_y) her 0.1 s'lik adimda
traci.vehicle.moveToXY ile dogrudan uygular (exact_replay_compare.py ile ayni fikir).
Boylece SUMO panelinde araclar, videodaki araclarla AYNI ANDA AYNI YERDEN gecer.

Iyilestirmeler (exact_replay_compare.py'ye gore):
  - Izler 0.1 s'lik duzenli zaman eksenine interpolasyonla oturtulur ve
    0.5 s'lik kayan ortalama ile yumusatilir (tespit titremesi azalir).
  - Aracin yonu hareket dogrultusundan hesaplanir (duran araclarda son yon korunur).
  - 1 s'den kisa (buyuk ihtimalle hatali) izler atlanir.
  - Kayit, record_demo.py ile ayni sekilde programatik yapilir (screenshot + ffmpeg).

Kullanim:
    python record_replay_demo.py \
        --video ../videos/Bellevue_116th_NE12th__2017-09-10_19-08-25.mp4 \
        --tracks ../tracks/2017-09-10_19_stitched_real.csv \
        --out bellevue_side_by_side_replay.mp4 --start auto --duration 90
"""
import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np
import pandas as pd

from record_demo import (BAR_H, PANEL_H, PANEL_W, fit_into, info_bar,
                         intersection_center, label, traci, wait_and_read)

STEP = 0.1
VTYPES = {  # cls -> (SUMO shape, uzunluk m, genislik m, renk)
    "car": ("passenger", 4.8, 1.8, (255, 210, 0)),
    "truck": ("truck", 8.0, 2.4, (255, 120, 0)),
    "bus": ("bus", 12.0, 2.5, (0, 170, 255)),
    "motorcycle": ("motorcycle", 2.2, 0.8, (255, 0, 200)),
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks", required=True, help="real_x/real_y iceren iz dosyasi")
    ap.add_argument("--net", default="../network/bellevue_116th_ne12th.net.xml")
    ap.add_argument("--poly", default="../network/bellevue_116th_ne12th.poly.xml")
    ap.add_argument("--gui-settings", default="../sumo/view_settings.xml")
    ap.add_argument("--zone-edge-map", default="../zone_to_edge.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="auto",
                    help="Baslangic saniyesi ya da 'auto' (en yogun aralik)")
    ap.add_argument("--duration", type=float, default=90.0)
    ap.add_argument("--half-width", type=float, default=100.0)
    ap.add_argument("--min-track-s", type=float, default=1.0)
    ap.add_argument("--smooth", type=int, default=5, help="Kayan ortalama penceresi (ornek)")
    ap.add_argument("--window-size", default="1400,900")
    ap.add_argument("--crf", type=int, default=20)
    return ap.parse_args()


def prepare_tracks(path, min_track_s, smooth):
    """Her izi 0.1 s'lik eksene oturt, yumusat, yon hesapla.
    Donus: {zaman_adimi(int): [(track_id, cls, x, y, aci), ...]}"""
    df = pd.read_csv(path, usecols=["time_s", "track_id", "cls", "real_x", "real_y"])
    by_step = {}
    n_used = 0
    for tid, g in df.groupby("track_id"):
        g = g.sort_values("time_s")
        k = np.round(g["time_s"].to_numpy() / STEP).astype(int)
        if (k[-1] - k[0]) * STEP < min_track_s:
            continue
        n_used += 1
        s = pd.DataFrame({"x": g["real_x"].to_numpy(), "y": g["real_y"].to_numpy()}, index=k)
        s = s[~s.index.duplicated()].reindex(range(k[0], k[-1] + 1)).interpolate()
        s = s.rolling(smooth, center=True, min_periods=1).mean()
        xs, ys = s["x"].to_numpy(), s["y"].to_numpy()
        cls = g["cls"].mode().iloc[0]
        # Baslangic yonu: izin toplam yer degistirmesi (yeterince hareket ettiyse)
        dx, dy = xs[-1] - xs[0], ys[-1] - ys[0]
        angle = math.degrees(math.atan2(dx, dy)) % 360 if math.hypot(dx, dy) > 2.0 else None
        half = 5  # +-0.5 s'lik farktan anlik yon
        for i, step in enumerate(s.index):
            a, b = max(0, i - half), min(len(xs) - 1, i + half)
            ddx, ddy = xs[b] - xs[a], ys[b] - ys[a]
            if math.hypot(ddx, ddy) > 0.8:
                angle = math.degrees(math.atan2(ddx, ddy)) % 360
            by_step.setdefault(int(step), []).append((tid, cls, xs[i], ys[i], angle))
    return by_step, n_used


def busiest_start(by_step, duration, cx, cy, hw, hh, min_start=3, search_until=3600):
    per_s = {}
    for step, items in by_step.items():
        if step % 10 == 0:
            per_s[step // 10] = sum(1 for _, _, x, y, _ in items
                                    if abs(x - cx) <= hw and abs(y - cy) <= hh)
    dur = int(duration)
    counts = [per_s.get(s, 0) for s in range(int(search_until) + dur)]
    best, best_score = 0, -1
    for s in range(int(search_until) - dur):
        if counts[s] < min_start:
            continue
        score = sum(counts[s:s + dur])
        if score > best_score:
            best, best_score = s, score
    print(f">>> En yogun aralik: {best}-{best + dur}s (baslangicta {counts[best]} arac, "
          f"ortalama {best_score / dur:.1f})")
    return float(best)


def main():
    args = parse_args()

    t0 = time.time()
    by_step, n_used = prepare_tracks(args.tracks, args.min_track_s, args.smooth)
    print(f">>> {n_used} iz hazirlandi ({time.time() - t0:.0f}s)")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"HATA: video acilamadi: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    sumo_cmd = ["sumo-gui", "-n", args.net, "-a", args.poly,
                "--gui-settings-file", args.gui_settings,
                "--start", "--quit-on-end", "--step-length", str(STEP),
                "--window-size", args.window_size, "--window-pos", "0,0",
                "--delay", "0", "--no-step-log", "true", "--no-warnings", "true",
                "--collision.action", "none", "--time-to-teleport", "-1"]
    traci.start(sumo_cmd, stdout=subprocess.DEVNULL)
    view_id = traci.gui.getIDList()[0]

    # Yer tutucu rota: en uzun normal edge (exact_replay_compare.py ile ayni)
    edges = [e for e in traci.edge.getIDList() if not e.startswith(":")]
    placeholder = max(edges, key=lambda e: traci.lane.getLength(f"{e}_0"))
    traci.route.add("dummy_route", [placeholder])
    for cls, (shape, length, width, color) in VTYPES.items():
        traci.vehicletype.copy("DEFAULT_VEHTYPE", f"replay_{cls}")
        traci.vehicletype.setShapeClass(f"replay_{cls}", shape)
        traci.vehicletype.setLength(f"replay_{cls}", length)
        traci.vehicletype.setWidth(f"replay_{cls}", width)
        traci.vehicletype.setColor(f"replay_{cls}", color + (255,))

    # Gorunum: record_demo.py ile ayni (ilk adimdan sonra ayarlanir)
    traci.simulationStep()
    cx, cy = intersection_center(args.zone_edge_map)
    hw = args.half_width
    traci.gui.setBoundary(view_id, cx - hw, cy - hw * 9 / 16, cx + hw, cy + hw * 9 / 16)
    traci.gui.setOffset(view_id, cx, cy)
    traci.simulationStep()
    (x0, _), (x1, _) = traci.gui.getBoundary(view_id)
    traci.gui.setZoom(view_id, traci.gui.getZoom(view_id) * (x1 - x0) / (2 * hw))
    traci.simulationStep()

    start = (busiest_start(by_step, args.duration, cx, cy, hw, hw * 9 / 16)
             if args.start == "auto" else float(args.start))
    n_frames = int(round(args.duration * fps))

    # Video: cap.set bu videoda kayiyor (bkz. record_demo.py) - kareleri sirayla atla
    for _ in range(int(round(start * fps))):
        if not cap.grab():
            sys.exit("HATA: video, baslangic saniyesinden once bitti.")

    # Simulasyon saati izlerden bagimsiz; izlerin zamanini simulasyon adimindan
    # ayri sayiyoruz: rep_step = gosterilen iz zamani / 0.1
    rep_step = int(round(start / STEP)) - 1
    active = {}  # track_id -> sumo arac ID'si
    reinserts = 0

    def apply_step(k):
        """k. iz adimindaki araclari ekle/sil/yerlestir (bir sonraki simulationStep'te gecerli)."""
        nonlocal reinserts
        items = {tid: (cls, x, y, a) for tid, cls, x, y, a in by_step.get(k, [])}
        for tid in list(active):
            if tid not in items:
                try:
                    traci.vehicle.remove(active.pop(tid))
                except traci.exceptions.TraCIException:
                    pass
        known = set(traci.vehicle.getIDList()) | set(traci.simulation.getPendingVehicles())
        for tid, (cls, x, y, a) in items.items():
            vid = active.get(tid)
            if vid is not None and vid not in known:
                vid = None  # SUMO araci dusurmus - yeni ID ile tekrar ekle
                reinserts += 1
            if vid is None:
                vid = f"v{tid}_{k}"
                traci.vehicle.add(vid, "dummy_route", typeID=f"replay_{cls}"
                                  if cls in VTYPES else "replay_car", depart="now")
                traci.vehicle.setSpeedMode(vid, 0)
                traci.vehicle.setSpeed(vid, 0)
                active[tid] = vid
            traci.vehicle.moveToXY(vid, "", 0, x, y,
                                   angle=a if a is not None else traci.constants.INVALID_DOUBLE_VALUE,
                                   keepRoute=2)

    shot_dir = tempfile.mkdtemp(prefix="sumo_replay_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
    out_w, out_h = 2 * PANEL_W, PANEL_H + BAR_H
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}",
         "-r", f"{fps:.6f}", "-i", "-",
         "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf),
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out],
        stdin=subprocess.PIPE)

    def advance():
        """Bir iz adimi ilerle, screenshot al. Donus: (goruntu ya da None)."""
        nonlocal rep_step
        rep_step += 1
        apply_step(rep_step)
        path = os.path.join(shot_dir, f"s{rep_step:07d}.png")
        traci.gui.screenshot(view_id, path)
        traci.simulationStep()
        img = wait_and_read(path)
        if img is not None:
            os.remove(path)
        return img

    sim_img = advance()
    missing = 0
    t_wall = time.time()
    print(f">>> {n_frames} kare kaydedilecek ({start:.0f}s -> {start + args.duration:.0f}s)")
    try:
        for i in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                print(">>> Video bitti.")
                break
            video_t = start + i / fps
            while rep_step * STEP < video_t - 1e-6:
                img = advance()
                if img is not None:
                    sim_img = img
                else:
                    missing += 1
            rep_t = rep_step * STEP

            left = fit_into(frame, PANEL_W, PANEL_H)
            right = fit_into(sim_img, PANEL_W, PANEL_H) if sim_img is not None \
                else np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
            label(left, "GERCEK VIDEO")
            label(right, "DIJITAL IKIZ (SUMO) - videodaki izlerin birebir tekrari")
            combined = np.vstack([info_bar(out_w, video_t, rep_t, len(active)),
                                  np.hstack([left, right])])
            ffmpeg.stdin.write(combined.tobytes())
            if i % int(fps * 10) == 0:
                print(f"    kare {i}/{n_frames}  video t={video_t:6.1f}s  iz t={rep_t:6.1f}s  "
                      f"arac={len(active)}  ({time.time() - t_wall:.0f}s gecti)")
    finally:
        ffmpeg.stdin.close()
        ffmpeg.wait()
        cap.release()
        traci.close()
        shutil.rmtree(shot_dir, ignore_errors=True)

    if missing:
        print(f"[uyari] {missing} SUMO screenshot'i alinamadi (onceki kare tekrarlandi).")
    if reinserts:
        print(f"[bilgi] {reinserts} kez SUMO'nun dusurdugu arac yeniden eklendi.")
    print(f">>> Kaydedildi: {args.out}")


if __name__ == "__main__":
    main()
