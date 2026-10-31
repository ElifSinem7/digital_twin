#!/usr/bin/env python3
"""
sync_compare.py
================
Gercek trafik kamerasi videosu ile SUMO dijital ikizini YAN YANA,
SENKRON olarak calistirip karsilastiran arac.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("HATA: SUMO_HOME ortam degiskeni tanimli degil.")

import traci  # noqa: E402


APPROACH_EDGES = {
    "Kuzey (116th Ave NE)": "428067750#0",
    "Guney (116th Ave NE)": "-428067754#0",
    "Dogu (NE 12th St)": "428067759#0",
    "Bati (NE 12th St)": "428067756",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--sumocfg", required=True)
    p.add_argument("--sumo-binary", default="sumo-gui")
    p.add_argument("--video-start-offset", type=float, default=0.0)
    p.add_argument("--output", default=None)
    p.add_argument("--live", action="store_true")
    p.add_argument("--panel-height", type=int, default=540)
    p.add_argument("--screenshot-dir", default="/dev/shm/sumo_screenshots")
    return p.parse_args()


def get_vehicle_counts():
    counts = {}
    for label, edge_id in APPROACH_EDGES.items():
        try:
            counts[label] = traci.edge.getLastStepVehicleNumber(edge_id)
        except traci.exceptions.TraCIException:
            counts[label] = None
    return counts


def draw_info_bar(frame, sim_time, counts, width, bar_height=70):
    bar = np.zeros((bar_height, width, 3), dtype=np.uint8)
    text = f"Sim. zamani: {sim_time:6.1f} s"
    cv2.putText(bar, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)
    x = 10
    y = 55
    for label, n in counts.items():
        txt = f"{label}: {n if n is not None else '-'}"
        cv2.putText(bar, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)
        x += 260
    return np.vstack([bar, frame])


def resize_to_height(img, height):
    h, w = img.shape[:2]
    scale = height / h
    return cv2.resize(img, (int(w * scale), height))


def main():
    args = parse_args()
    os.makedirs(args.screenshot_dir, exist_ok=True)
    print(f">>> Ekran goruntusu klasoru: {args.screenshot_dir}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"HATA: Video acilamadi: {args.video}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if args.video_start_offset > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.video_start_offset * 1000)

    print(f">>> Video FPS: {video_fps:.2f}")
    print(">>> SUMO baslatiliyor...")

    sumo_cmd = [args.sumo_binary, "-c", args.sumocfg, "--start", "--quit-on-end"]
    traci.start(sumo_cmd)

    try:
        view_ids = traci.gui.getIDList()
        if view_ids:
            traci.gui.setBoundary(view_ids[0], 500, 700, 700, 900)
            print(f">>> Zoom ayarlandi: {view_ids[0]}")
    except traci.exceptions.TraCIException as e:
        print(f"[uyari] zoom ayarlanamadi: {e}")

    step_length = traci.simulation.getDeltaT()
    sim_steps_per_video_frame = max(1, round((1.0 / video_fps) / step_length))

    writer = None
    frame_idx = 0

    try:
        while True:
            ret, video_frame = cap.read()
            if not ret:
                print(">>> Video bitti.")
                break

            if traci.simulation.getMinExpectedNumber() <= 0:
                print(">>> Simulasyonda arac kalmadi, sonlandiriliyor.")
                break

            for _ in range(sim_steps_per_video_frame):
                traci.simulationStep()

            sim_time = traci.simulation.getTime()
            counts = get_vehicle_counts()

            shot_path = os.path.join(args.screenshot_dir, f"f{frame_idx:06d}.png")
            sim_frame = None
            try:
                traci.gui.screenshot("View #0", shot_path)
                ok = False
                for _ in range(150):
                    if os.path.exists(shot_path) and os.path.getsize(shot_path) > 0:
                        ok = True
                        break
                    time.sleep(0.02)
                if not ok and frame_idx < 10:
                    print(f"[DEBUG] kare {frame_idx}: dosya olusmadi. "
                          f"dizin var mi: {os.path.isdir(args.screenshot_dir)}")
                sim_frame = cv2.imread(shot_path) if ok else None
            except traci.exceptions.TraCIException as e:
                if frame_idx < 10:
                    print(f"[DEBUG] kare {frame_idx}: TraCIException: {e}")

            if sim_frame is None:
                sim_frame = np.zeros_like(video_frame)
                cv2.putText(sim_frame, "SUMO goruntusu alinamadi",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            video_r = resize_to_height(video_frame, args.panel_height)
            sim_r = resize_to_height(sim_frame, args.panel_height)
            cv2.putText(video_r, "GERCEK VIDEO", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(sim_r, "DIJITAL IKIZ (SUMO)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            combined = cv2.hconcat([video_r, sim_r])
            combined = draw_info_bar(combined, sim_time, counts, combined.shape[1])

            if args.output:
                if writer is None:
                    h, w = combined.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.output, fourcc, video_fps, (w, h))
                writer.write(combined)

            if args.live:
                cv2.imshow("Gercek Video vs Dijital Ikiz", combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print(">>> Kullanici tarafindan durduruldu.")
                    break

            try:
                os.remove(shot_path)
            except OSError:
                pass

            frame_idx += 1

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.live:
            cv2.destroyAllWindows()
        traci.close()
        try:
            os.rmdir(args.screenshot_dir)
        except OSError:
            pass

    print(">>> Bitti.")
    if args.output:
        print(f">>> Karsilastirma videosu kaydedildi: {args.output}")


if __name__ == "__main__":
    main()
