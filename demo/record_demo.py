#!/usr/bin/env python3
"""
record_demo.py
================
live_dual_window.py ile AYNI simulasyonu (ayni sumocfg, ayni zoom, ayni
"video zamani = simulasyon zamani" senkron mantigi) calistirir, ama ekran
kaydi yerine PROGRAMATIK kayit yapar:

  1) Her SUMO adiminda (0.1 s) traci.gui.screenshot ile SUMO gorunumunu alir.
  2) Kaynak videonun her karesini, o anki simulasyon karesiyle yan yana koyar.
  3) Birlesik kareleri dogrudan ffmpeg'e (libx264, H.264) aktarir.

Not: traci.gui.screenshot, istegi bir SONRAKI simulationStep() sonunda
yerine getirir. Bu yuzden once screenshot istenir, sonra adim atilir ve
dosyanin diske yazilmasi beklenir - dosya adim SONRASI durumu gosterir.

Kullanim (demo/make_demo.sh bunu sizin icin yapar):
    python record_demo.py \
        --video ../videos/Bellevue_116th_NE12th__2017-09-10_19-08-25.mp4 \
        --sumocfg ../sumo/digital_twin.sumocfg \
        --out bellevue_side_by_side.mp4 --start 0 --duration 90
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("HATA: SUMO_HOME tanimli degil.")

import traci  # noqa: E402

PANEL_W, PANEL_H = 960, 540
BAR_H = 60
# Video dosya adindaki baslangic saati (2017-09-10 19:08:25)
VIDEO_CLOCK_START = datetime.datetime(2017, 9, 10, 19, 8, 25)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--sumocfg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=float, default=0.0,
                    help="Kaydin baslayacagi video/simulasyon saniyesi")
    ap.add_argument("--duration", type=float, default=90.0,
                    help="Kayit suresi (saniye)")
    ap.add_argument("--window-size", default="1400,900",
                    help="sumo-gui pencere boyutu (screenshot cozunurlugunu belirler)")
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--view", choices=["intersection", "live"], default="intersection",
                    help="intersection: kavsaga yakinlas; live: live_dual_window.py'deki zoom")
    ap.add_argument("--zone-edge-map", default="../zone_to_edge.json",
                    help="Kavsak merkezini bulmak icin giris edge'leri (--view intersection)")
    ap.add_argument("--half-width", type=float, default=100.0,
                    help="Kavsak merkezinden gosterilecek yatay yari genislik (m)")
    return ap.parse_args()


def intersection_center(zone_edge_map):
    """Giris edge'lerinin bittigi dugumlerin ortalamasi = kamera kavsaginin merkezi."""
    with open(zone_edge_map) as f:
        zone_edge = json.load(f)
    pts = []
    for zone, edge in zone_edge.items():
        if zone.startswith("entry"):
            shape = traci.lane.getShape(f"{edge}_0")
            pts.append(shape[-1])
    return float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))


def fit_into(img, w, h, bg=(30, 30, 30)):
    """Oran korunarak w x h kutuya sigdir, kalan alani koyu renkle doldur."""
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = int(round(iw * scale)), int(round(ih * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), bg, dtype=np.uint8)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def label(img, text, org=(12, 34)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)


def info_bar(width, video_t, sim_t, n_veh):
    bar = np.full((BAR_H, width, 3), (20, 20, 20), dtype=np.uint8)
    cv2.putText(bar, "Bellevue 116th Ave NE & NE 12th St  -  Gercek video vs SUMO dijital ikiz",
                (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    clock = (VIDEO_CLOCK_START + datetime.timedelta(seconds=video_t)).strftime("%H:%M:%S")
    txt = f"Video {clock} (t={video_t:6.1f}s)   SUMO t={sim_t:6.1f}s   Arac: {n_veh}"
    cv2.putText(bar, txt, (width - 760, 38), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 255), 2, cv2.LINE_AA)
    return bar


def wait_and_read(path, timeout=10.0):
    """sumo-gui screenshot'i asenkron yazar - dosya tam okunabilene kadar bekle."""
    t_end = time.time() + timeout
    while time.time() < t_end:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            img = cv2.imread(path)
            if img is not None:
                return img
        time.sleep(0.01)
    return None


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"HATA: video acilamadi: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(round(args.duration * fps))
    # cap.set(CAP_PROP_POS_FRAMES) bu videoda (degisken kare hizi) ~142 kare ERKEN
    # bir kareye dusuyor (399. saniyede olculdu). Arac zamanlari (time_s) kareler
    # sirayla okunarak hesaplandigi icin, baslangica kadar kareleri sirayla atliyoruz.
    for _ in range(int(round(args.start * fps))):
        if not cap.grab():
            sys.exit("HATA: video, baslangic saniyesinden once bitti.")

    shot_dir = tempfile.mkdtemp(prefix="sumo_demo_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
    print(f">>> Video FPS: {fps:.3f}, {n_frames} kare kaydedilecek ({args.start}s -> {args.start + args.duration}s)")
    print(f">>> Gecici screenshot klasoru: {shot_dir}")

    sumo_cmd = ["sumo-gui", "-c", args.sumocfg, "--start", "--quit-on-end",
                "--window-size", args.window_size, "--window-pos", "0,0",
                "--delay", "0", "--no-step-log", "true"]
    traci.start(sumo_cmd, stdout=subprocess.DEVNULL)
    view_id = traci.gui.getIDList()[0]
    # sumo-gui, view_settings.xml'deki viewport'u GUI yuklenirken uyguluyor ve
    # bu, simulasyon baslamadan yapilan TraCI zoom ayarlarinin uzerine yazabiliyor.
    # Bu yuzden gorunumu ilk adimdan SONRA ayarliyoruz.
    traci.simulationStep()
    if args.view == "live":
        # live_dual_window.py ile ayni gorunum
        traci.gui.setBoundary(view_id, 500, 700, 700, 900)
    else:
        cx, cy = intersection_center(args.zone_edge_map)
        hw = args.half_width
        traci.gui.setBoundary(view_id, cx - hw, cy - hw * 9 / 16, cx + hw, cy + hw * 9 / 16)
        traci.gui.setOffset(view_id, cx, cy)
        traci.simulationStep()
        # setOffset zoom'u bozabiliyor - gorunen genisligi olcup zoom'u duzelt
        (x0, _), (x1, _) = traci.gui.getBoundary(view_id)
        traci.gui.setZoom(view_id, traci.gui.getZoom(view_id) * (x1 - x0) / (2 * hw))
        print(f">>> Kavsak merkezi: ({cx:.1f}, {cy:.1f}), +-{hw:.0f} m")
    traci.simulationStep()
    print(f">>> Gorunum: zoom={traci.gui.getZoom(view_id):.1f}  "
          f"offset={traci.gui.getOffset(view_id)}  boundary={traci.gui.getBoundary(view_id)}")

    # Baslangica kadar screenshot almadan hizli ilerlet
    while traci.simulation.getTime() < args.start - 1e-6:
        traci.simulationStep()

    out_w, out_h = 2 * PANEL_W, PANEL_H + BAR_H
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}",
         "-r", f"{fps:.6f}", "-i", "-",
         "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf),
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out],
        stdin=subprocess.PIPE)

    # Ilk SUMO karesi: mevcut durumu gormek icin bir adim at
    shot_idx = 0
    path = os.path.join(shot_dir, f"s{shot_idx:06d}.png")
    traci.gui.screenshot(view_id, path)
    traci.simulationStep()
    sim_img = wait_and_read(path)
    os.remove(path)
    sim_t = traci.simulation.getTime()
    missing = 0
    t_wall = time.time()

    try:
        for i in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                print(">>> Video bitti.")
                break
            video_t = args.start + i / fps

            # Simulasyonu video zamanina yetistir (live_dual_window.py ile ayni mantik)
            while sim_t < video_t - 1e-6:
                shot_idx += 1
                path = os.path.join(shot_dir, f"s{shot_idx:06d}.png")
                traci.gui.screenshot(view_id, path)
                traci.simulationStep()
                img = wait_and_read(path)
                if img is not None:
                    sim_img = img
                    os.remove(path)
                else:
                    missing += 1
                sim_t = traci.simulation.getTime()

            left = fit_into(frame, PANEL_W, PANEL_H)
            right = fit_into(sim_img, PANEL_W, PANEL_H) if sim_img is not None \
                else np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
            label(left, "GERCEK VIDEO")
            label(right, "DIJITAL IKIZ (SUMO)")
            combined = np.vstack([info_bar(out_w, video_t, sim_t, traci.vehicle.getIDCount()),
                                  np.hstack([left, right])])
            ffmpeg.stdin.write(combined.tobytes())

            if i % int(fps * 5) == 0:
                print(f"    kare {i}/{n_frames}  video t={video_t:6.1f}s  sim t={sim_t:6.1f}s  "
                      f"({time.time() - t_wall:.0f}s gecti)")
    finally:
        ffmpeg.stdin.close()
        ffmpeg.wait()
        cap.release()
        traci.close()
        shutil.rmtree(shot_dir, ignore_errors=True)

    if missing:
        print(f"[uyari] {missing} SUMO screenshot'i alinamadi (onceki kare tekrarlandi).")
    print(f">>> Kaydedildi: {args.out}")


if __name__ == "__main__":
    main()
