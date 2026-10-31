#!/usr/bin/env python3
"""
record_mirror_demo.py
=======================
AYNA modu yan yana demo: videodaki her arac, her 0.1 s'de gozlenen yerine
traci.vehicle.moveToXY(keepRoute=2) ile konur (rota bagimliligi yok -> 0.2 m'lik kuzey
giris kenari sorun olmaz); iz bitince ya da gecerli bolgeden cikinca silinir.
Trafik isiklari her adimda signal_plan_*.csv'den setRedYellowGreenState ile uygulanir.

Gorunum: sumo-gui kamera bakis yonune dondurulur (yukari = kameranin baktigi yon),
kameranin gecerli gorus alanina yakinlastirilir ve bu alan SUMO'da cizilir.
Araclar 1.3x buyutulur ve GELDIKLERI YONE gore renklenir; ayni renk videodaki kutuda da kullanilir.

Kullanim:
    python record_mirror_demo.py --start 399 --duration 90 --out ../demo/bellevue_side_by_side_v2.mp4
"""
import argparse
import datetime
import json
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

from fisheye_map import FisheyeGroundMap
from view_utils import camera_bearing, polygon_to_image, visible_polygon

if "SUMO_HOME" not in os.environ:
    sys.exit("HATA: SUMO_HOME tanimli degil.")
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TLS = "cluster_53190763_5896114911_7619513267_7619513268_#3more"
STEP = 0.1
PANEL_W, PANEL_H, BAR_H, SIG_H = 960, 540, 56, 64
VIDEO_CLOCK_START = datetime.datetime(2017, 9, 10, 19, 8, 25)
# geldigi yon -> renk (RGB); kirmizi/sari/yesil isiklarla karismasin diye secildi
APPROACH_RGB = {"N": (30, 144, 255), "E": (255, 0, 255), "S": (0, 206, 209), "W": (255, 140, 0), "?": (190, 190, 190)}
APPROACH_TR = {"N": "kuzey", "E": "dogu", "S": "guney", "W": "bati", "?": "bilinmiyor"}
VTYPES = {"car": ("passenger", 4.8, 1.9), "truck": ("truck", 8.0, 2.5),
          "bus": ("bus", 12.0, 2.6), "motorcycle": ("motorcycle", 2.2, 0.9)}
SIG_BGR = {"G": (0, 200, 0), "g": (0, 200, 0), "F": (0, 170, 255), "Y": (0, 220, 255), "y": (0, 220, 255),
           "R": (0, 0, 230), "r": (0, 0, 230), "-": (90, 90, 90)}


def bgr(rgb):
    return (rgb[2], rgb[1], rgb[0])


def fit_into(img, w, h, bg=(30, 30, 30)):
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = int(round(iw * s)), int(round(ih * s))
    out = np.full((h, w, 3), bg, np.uint8)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out, s, x0, y0


def wait_and_read(path, timeout=10.0):
    t_end = time.time() + timeout
    while time.time() < t_end:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            img = cv2.imread(path)
            if img is not None:
                return img
        time.sleep(0.01)
    return None


def text(img, s, org, scale=0.6, col=(255, 255, 255), th=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(ROOT, "videos", "Bellevue_116th_NE12th__2017-09-10_19-08-25.mp4"))
    ap.add_argument("--tracks", default=os.path.join(HERE, "mirror_tracks_384_504.csv"))
    ap.add_argument("--ids", default=os.path.join(HERE, "mirror_track_ids.json"))
    ap.add_argument("--boxes", default=os.path.join(ROOT, "tracks", "window_384_504_boxes.csv"))
    ap.add_argument("--plan", default=os.path.join(HERE, "signal_plan_384_504.csv"))
    ap.add_argument("--mapping", default=os.path.join(HERE, "fisheye_mapping.json"))
    ap.add_argument("--net", default=os.path.join(ROOT, "network", "bellevue_116th_ne12th.net.xml"))
    ap.add_argument("--poly", default=os.path.join(ROOT, "network", "bellevue_116th_ne12th.poly.xml"))
    ap.add_argument("--gui-settings", default=os.path.join(ROOT, "sumo", "view_settings.xml"))
    ap.add_argument("--start", type=float, default=399.0)
    ap.add_argument("--duration", type=float, default=90.0)
    ap.add_argument("--max-dist", type=float, default=40.0)
    ap.add_argument("--size-scale", type=float, default=1.3, help="Arac boyutu carpani (gorunurluk icin)")
    ap.add_argument("--window-size", default="1600,960")
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(ROOT, "demo", "bellevue_side_by_side_v2.mp4"))
    ap.add_argument("--log", default=os.path.join(HERE, "mirror_run_log.csv"))
    args = ap.parse_args()

    _m = __import__("re").search(r"_(v\d+)\.mp4$", args.out)
    version = _m.group(1) if _m else "ayna"
    tracks = pd.read_csv(args.tracks)
    ids0 ={int(k): v for k, v in json.load(open(args.ids)).items()}
    # Arac SUMO'da ancak videodaki ilk tespiti gelince belirsin / son tespitinden sonra kalmasin:
    # 0.1 s'lik ize oturtma ve SUMO adiminin video karesinin onunde gitmesi araci ~0.1-0.3 s erken gosteriyordu.
    bb = pd.read_csv(args.boxes, usecols=["time_s", "track_id"])
    bb["root"] = [ids0.get(int(t), {"root": int(t)})["root"] for t in bb.track_id]
    det_span = bb.groupby("root").time_s.agg(["min", "max"])
    lo = tracks.track_id.map(det_span["min"]); hi = tracks.track_id.map(det_span["max"])
    show_t = tracks.step * STEP - STEP   # bu adim, video zamani (k-1)*0.1 < t <= k*0.1 iken gosterilir
    tracks = tracks[(show_t >= lo - 1e-6) & (show_t <= hi + 1e-6)]
    by_step = {k: g for k, g in tracks.groupby("step")}
    origin_of = tracks.drop_duplicates("track_id").set_index("track_id").origin.to_dict()
    ids = {int(k): v for k, v in json.load(open(args.ids)).items()}
    boxes = pd.read_csv(args.boxes)
    boxes_by_frame = {f: g for f, g in boxes.groupby("frame")}
    plan = pd.read_csv(args.plan)
    plan_by_k = dict(zip(np.round(plan.time_s / STEP).astype(int), plan.itertuples(index=False)))
    m = FisheyeGroundMap.load(args.mapping)
    region = visible_polygon(m, args.max_dist)
    region_img = polygon_to_image(m, region)
    region_img = np.int32(region_img[np.isfinite(region_img).all(1)])
    bearing = camera_bearing(m)
    smask_path = os.path.join(HERE, "static_mask.png")
    mask_contours = []
    if os.path.exists(smask_path):
        sm = cv2.imread(smask_path, cv2.IMREAD_GRAYSCALE)
        mask_contours, _ = cv2.findContours((sm > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    last_box = {}   # ayna izi -> (kare, kutu): kisa tespit kayiplarinda kesikli kutu cizmek icin
    GHOST_FRAMES = 30  # ~1 s (prepare_mirror'daki <= 1 s bosluk doldurma siniri)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    first_frame = int(round(args.start * fps))
    for _ in range(first_frame):   # cap.set bu videoda kayiyor: sirayla atla
        cap.grab()
    n_frames = int(round(args.duration * fps))

    sumo_cmd = ["sumo-gui", "-n", args.net, "-a", args.poly, "--gui-settings-file", args.gui_settings,
                "--begin", f"{args.start - 1:.1f}", "--step-length", str(STEP), "--start", "--quit-on-end",
                "--window-size", args.window_size, "--window-pos", "0,0", "--delay", "0",
                "--no-step-log", "true", "--no-warnings", "true",
                "--collision.action", "none", "--time-to-teleport", "-1"]
    traci.start(sumo_cmd, stdout=subprocess.DEVNULL)
    view = traci.gui.getIDList()[0]
    edges = [e for e in traci.edge.getIDList() if not e.startswith(":")]
    placeholder = max(edges, key=lambda e: traci.lane.getLength(f"{e}_0"))
    traci.route.add("mirror_route", [placeholder])
    for cls, (shape, length, width) in VTYPES.items():
        tid = f"mirror_{cls}"
        traci.vehicletype.copy("DEFAULT_VEHTYPE", tid)
        traci.vehicletype.setShapeClass(tid, shape)
        traci.vehicletype.setLength(tid, length * args.size_scale)
        traci.vehicletype.setWidth(tid, width * args.size_scale)
        traci.vehicletype.setMinGap(tid, 0.0)
    traci.polygon.add("kamera_gorus_alani", [tuple(p) for p in np.array(region.exterior.coords)],
                      (0, 255, 255, 255), fill=False, layer=50, lineWidth=0.35)
    traci.trafficlight.setRedYellowGreenState(TLS, plan.state.iloc[0])

    # ---- gorunum: kamera yonune dondur, gorus alanina sigdir ----
    traci.simulationStep()
    cx, cy = region.centroid.coords[0]
    traci.gui.setAngle(view, bearing)
    traci.gui.setOffset(view, cx, cy)
    traci.gui.setZoom(view, 400)
    # olcek kalibrasyonu: ekranda yatay dogrultuda aralari 30 m olan iki MOR isaret
    # (kirmizi kullanma: SUMO'nun stop cizgisindeki kirmizi isik cubuklari karisiyor)
    th = math.radians(bearing)
    ex = np.array([math.cos(th), -math.sin(th)])  # ekranda saga dogru dunya yonu
    ey = np.array([math.sin(th), math.cos(th)])   # ekranda yukari = kamera bakis yonu
    SEP = 30.0
    for nm, s in (("cal_a", -SEP / 2), ("cal_b", SEP / 2)):
        p = np.array([cx, cy]) + ex * s
        traci.polygon.add(nm, [tuple(p + d) for d in ((-1, -1), (1, -1), (1, 1), (-1, 1))],
                          (255, 0, 255, 255), fill=True, layer=60)
    shot_dir = tempfile.mkdtemp(prefix="sumo_mirror_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
    R = np.array(region.exterior.coords) - [cx, cy]
    w_m = np.ptp(R @ ex) * 1.06; h_m = np.ptp(R @ ey) * 1.06

    def measure(tag):
        traci.simulationStep(); traci.simulationStep()
        path = os.path.join(shot_dir, f"cal_{tag}.png")
        traci.gui.screenshot(view, path); traci.simulationStep()
        img = wait_and_read(path)
        mask = np.all(np.abs(img.astype(int) - (255, 0, 255)) < 25, axis=2)
        ys, xs = np.nonzero(mask)
        mid = img.shape[1] / 2
        l, r = xs[xs < mid], xs[xs >= mid]
        if len(l) < 3 or len(r) < 3:
            return None, img.shape
        return (r.mean() - l.mean()) / SEP, img.shape

    zoom = 400.0
    for it in range(3):
        traci.gui.setZoom(view, zoom)
        ppm, shp = measure(it)
        if ppm is None:          # isaretler ekran disinda: uzaklas
            zoom /= 3; continue
        need = min(shp[1] / w_m, shp[0] / h_m)   # istenen piksel/metre
        print(f"    kalibrasyon {it}: zoom {zoom:.0f} -> {ppm:.2f} px/m, istenen {need:.2f} px/m")
        if abs(ppm / need - 1) < 0.03:
            break
        zoom *= need / ppm
    traci.gui.setZoom(view, zoom)
    for nm in ("cal_a", "cal_b"):
        traci.polygon.remove(nm)
    ci = np.zeros(shp, np.uint8)
    print(f">>> Kamera bakis yonu {bearing:.1f} deg -> SUMO gorunum acisi {bearing:.1f}; "
          f"gorus alani {w_m:.0f} x {h_m:.0f} m, zoom {zoom:.0f} (ekran {ci.shape[1]}x{ci.shape[0]})")

    while traci.simulation.getTime() < args.start - STEP / 2:
        traci.simulationStep()

    active, log, n_readd = {}, [], 0
    tls_samples, frame_log = [], []

    def apply_step(k):
        nonlocal n_readd
        rows = by_step.get(k)
        cur = {} if rows is None else {int(r.track_id): r for r in rows.itertuples(index=False)}
        for tid in list(active):
            if tid not in cur:
                try:
                    traci.vehicle.remove(active.pop(tid))
                except traci.exceptions.TraCIException:
                    active.pop(tid, None)
        known = set(traci.vehicle.getIDList()) | set(traci.simulation.getPendingVehicles())
        for tid, r in cur.items():
            vid = active.get(tid)
            if vid is not None and vid not in known:
                vid = None; n_readd += 1
            if vid is None:
                vid = f"t{tid}_{k}"
                cls = r.cls if r.cls in VTYPES else "car"
                traci.vehicle.add(vid, "mirror_route", typeID=f"mirror_{cls}", depart="now")
                traci.vehicle.setSpeedMode(vid, 0)
                traci.vehicle.setSpeed(vid, 0)
                traci.vehicle.setColor(vid, APPROACH_RGB.get(r.origin, APPROACH_RGB["?"]) + (255,))
                active[tid] = vid
            traci.vehicle.moveToXY(vid, "", 0, float(r.x), float(r.y), angle=float(r.angle), keepRoute=2)
        p = plan_by_k.get(k)
        if p is not None:
            traci.trafficlight.setRedYellowGreenState(TLS, p.state)
        return p

    out_w, out_h = 2 * PANEL_W, PANEL_H + BAR_H + SIG_H
    ffmpeg = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                               "-s", f"{out_w}x{out_h}", "-r", f"{fps:.6f}", "-i", "-", "-c:v", "libx264",
                               "-preset", "medium", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
                               "-movflags", "+faststart", args.out], stdin=subprocess.PIPE)
    k = int(round(args.start / STEP)) - 1
    sim_img, cur_plan, missing = None, None, 0

    def advance():
        nonlocal k, cur_plan
        k += 1
        cur_plan = apply_step(k) or cur_plan
        path = os.path.join(shot_dir, f"s{k:07d}.png")
        traci.gui.screenshot(view, path)
        traci.simulationStep()
        for tid, vid in active.items():
            try:
                x, y = traci.vehicle.getPosition(vid)
                log.append((k, round(k * STEP, 1), tid, origin_of.get(tid, "?"), round(x, 2), round(y, 2)))
            except traci.exceptions.TraCIException:
                pass
        img = wait_and_read(path)
        if img is not None:
            os.remove(path)
        return img

    sim_img = advance()
    t_wall = time.time()
    try:
        for i in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                break
            fidx = first_frame + i
            video_t = fidx / fps
            while k * STEP < video_t - 1e-6:
                img = advance()
                if img is not None:
                    sim_img = img
                else:
                    missing += 1
            # --- sol panel: video + ayni renkte kutular ---
            vis = frame.copy()
            cv2.polylines(vis, [region_img], True, (255, 255, 0), 1, cv2.LINE_AA)
            if mask_contours:
                cv2.drawContours(vis, mask_contours, -1, (90, 90, 90), 1, cv2.LINE_AA)
            n_vid = n_ghost = 0
            b = boxes_by_frame.get(fidx)
            present = set(active)
            seen = set()
            if b is not None:
                for r in b.itertuples(index=False):
                    info = ids.get(int(r.track_id), {"root": -1, "status": "short"})
                    root = info["root"]
                    if root in present and info["status"] == "mirror":
                        # SUMO aracini sadece ayna izi olusturur/gunceller; bu kutu o izin parcasi
                        col = bgr(APPROACH_RGB.get(origin_of.get(root, "?"), APPROACH_RGB["?"]))
                        cv2.rectangle(vis, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), col, 2)
                        last_box[root] = (fidx, (r.x1, r.y1, r.x2, r.y2))
                        seen.add(root)
                        n_vid += 1
                    else:   # gri: ayna disi (kisa iz / 40 m disi / sabit) -> SUMO'ya hic girmez, sayilmaz
                        cv2.rectangle(vis, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), (200, 200, 200), 1)
            # tespiti bu karede kaybolan ama SUMO'da (<= 1 s) surdurulen araclar: kesikli kutu
            for root in present - seen:
                lb = last_box.get(root)
                if lb is None or fidx - lb[0] > GHOST_FRAMES:
                    continue
                x1, y1, x2, y2 = [int(v) for v in lb[1]]
                col = bgr(APPROACH_RGB.get(origin_of.get(root, "?"), APPROACH_RGB["?"]))
                for (pa, pb) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
                    n = max(2, int(np.hypot(pb[0] - pa[0], pb[1] - pa[1]) / 8))
                    for s in range(0, n, 2):
                        q0 = (int(pa[0] + (pb[0] - pa[0]) * s / n), int(pa[1] + (pb[1] - pa[1]) * s / n))
                        q1 = (int(pa[0] + (pb[0] - pa[0]) * (s + 1) / n), int(pa[1] + (pb[1] - pa[1]) * (s + 1) / n))
                        cv2.line(vis, q0, q1, col, 2)
                n_ghost += 1
            left, _, _, _ = fit_into(vis, PANEL_W, PANEL_H)
            right = fit_into(sim_img, PANEL_W, PANEL_H)[0] if sim_img is not None else np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
            text(left, "GERCEK VIDEO", (12, 30), 0.8, (0, 255, 0), 2)
            text(right, "DIJITAL IKIZ (SUMO) - ayna modu", (12, 30), 0.8, (0, 255, 0), 2)
            text(right, "yukari = kameranin baktigi yon (KD)", (12, 56), 0.5)
            # kuzey oku (SUMO paneli)
            nx, ny = -math.sin(math.radians(bearing)), -math.cos(math.radians(bearing))
            c0 = np.array([PANEL_W - 45, 75])
            cv2.arrowedLine(right, tuple(np.int32(c0 - 22 * np.array([nx, ny]))), tuple(np.int32(c0 + 22 * np.array([nx, ny]))),
                            (255, 255, 255), 2, tipLength=0.35)
            text(right, "N", tuple(np.int32(c0 + 34 * np.array([nx, ny]) - [6, -6])), 0.6)
            # --- ust bilgi cubugu ---
            bar = np.full((BAR_H, out_w, 3), 20, np.uint8)
            clock = (VIDEO_CLOCK_START + datetime.timedelta(seconds=video_t)).strftime("%H:%M:%S")
            cv2.putText(bar, f"Bellevue 116th Ave NE & NE 12th St  -  video vs SUMO ayna ({version})", (14, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            ghost_txt = f" (+{n_ghost} anlik kayip)" if n_ghost else ""
            cv2.putText(bar, f"Video {clock} (t={video_t:6.1f}s)  SUMO t={k * STEP:6.1f}s  arac: video {n_vid}{ghost_txt} / SUMO {len(active)}",
                        (out_w - 1010, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            frame_log.append((round(video_t, 3), k, n_vid, n_ghost, len(active), sorted(present - seen)))
            # --- alt sinyal cubugu ---
            sig = np.full((SIG_H, out_w, 3), 28, np.uint8)
            x = 14
            cv2.putText(sig, "Isiklar (duz akis):", (x, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            x += 175
            if cur_plan is not None:
                items = [("W", cur_plan.W_thru, "videodan okundu"), ("S", cur_plan.S_thru, "videodan okundu"),
                         ("E", cur_plan.E_thru, "W sola FYA kuralindan"), ("N", cur_plan.N_thru, "S sola FYA kuralindan")]
                for arm, st, src in items:
                    cv2.circle(sig, (x + 10, 21), 10, SIG_BGR.get(st, (90, 90, 90)), -1)
                    cv2.putText(sig, f"{APPROACH_TR[arm]}dan gelen", (x + 26, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                bgr(APPROACH_RGB[arm]), 1, cv2.LINE_AA)
                    cv2.putText(sig, src, (x + 26, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
                    x += 245
            cv2.putText(sig, "Renk = aracin geldigi yon (iki panelde ayni). Kesikli: tespit <=1 s kayip, SUMO surduruyor. "
                        "Gri kutu: ayna disi (kisa iz / 40 m disi). Koyu gri cizgi: sabit-nesne maskesi. Camgobegi: gecerli gorus alani.",
                        (14, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
            combined = np.vstack([bar, np.hstack([left, right]), sig])
            ffmpeg.stdin.write(combined.tobytes())
            # her 3 s'de bir: SUMO durumu vs video lambasi
            if abs(video_t - round(video_t)) < 0.5 / fps and int(round(video_t)) % 3 == 0 and cur_plan is not None:
                tls_samples.append((round(video_t), traci.trafficlight.getRedYellowGreenState(TLS),
                                    cur_plan.W_thru, cur_plan.S_thru))
            if i % int(fps * 10) == 0:
                print(f"    kare {i}/{n_frames} t={video_t:6.1f}s SUMO arac={len(active)} ({time.time() - t_wall:.0f}s)")
    finally:
        ffmpeg.stdin.close(); ffmpeg.wait()
        cap.release()
        traci.close()
        shutil.rmtree(shot_dir, ignore_errors=True)
    pd.DataFrame(log, columns=["step", "time_s", "track_id", "origin", "x", "y"]).to_csv(args.log, index=False)
    pd.DataFrame(tls_samples, columns=["time_s", "sumo_state", "video_W", "video_S"]).to_csv(
        args.log.replace(".csv", "_tls.csv"), index=False)
    pd.DataFrame(frame_log, columns=["video_t", "step", "video_boxes", "video_ghost", "sumo", "sumo_without_box"]).to_csv(
        args.log.replace(".csv", "_frames.csv"), index=False)
    print(f">>> Kaydedildi: {args.out}  (eksik screenshot {missing}, yeniden eklenen arac {n_readd})")
    print(f">>> Log: {args.log}")


if __name__ == "__main__":
    main()
