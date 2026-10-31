#!/usr/bin/env python3
"""
exact_replay_compare.py
=========================
sync_compare.py'den FARKLI olarak, SUMO'nun kendi rota/fizik motoruna
GUVENMEZ - videodaki her GERCEK aracin, apply_homography.py ile
hesaplanmis gercek (real_x, real_y) konumuna gore, TraCI ile HER ADIMDA
ELLE yerlestirilmesini ("kuklacilik"/puppeting) saglar.

Sonuc: SUMO'daki araclar, videodaki araclarla NEREDEYSE BIREBIR AYNI
konumda/zamanda goruntulenir - "istatistiksel olarak benzer trafik"
DEGIL, gercek trafigin dogrudan yeniden oynatilmasi (trajectory replay).

Kurulum:
    pip install opencv-python numpy pandas traci

Kullanim:
    python exact_replay_compare.py \
        --video ../videos/Bellevue_116th_NE12th__2017-09-10_19-08-25.mp4 \
        --tracks ../tracks/2017-09-10_19_real.csv \
        --net ../network/bellevue_116th_ne12th.net.xml \
        --poly ../network/bellevue_116th_ne12th.poly.xml \
        --gui-settings ../sumo/view_settings.xml \
        --output ../comparison_exact.mp4 \
        --live
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("HATA: SUMO_HOME ortam degiskeni tanimli degil.")

import traci  # noqa: E402
import sumolib  # noqa: E402


def resize_to_height(img, height):
    h, w = img.shape[:2]
    scale = height / h
    return cv2.resize(img, (int(w * scale), height))


def draw_info_bar(frame, sim_time, n_active, width, bar_height=50):
    bar = np.zeros((bar_height, width, 3), dtype=np.uint8)
    text = f"Sim. zamani: {sim_time:6.1f} s   |   Aktif (gercek) arac: {n_active}"
    cv2.putText(bar, text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, frame])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks", required=True,
                     help="apply_homography.py ciktisi (real_x/real_y sutunlari sart)")
    ap.add_argument("--net", required=True)
    ap.add_argument("--poly", default=None)
    ap.add_argument("--gui-settings", default=None)
    ap.add_argument("--sumo-binary", default="sumo-gui")
    ap.add_argument("--output", default=None)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--panel-height", type=int, default=540)
    ap.add_argument("--time-offset", type=float, default=0.0,
                     help="Videonun hangi saniyesi simulasyonun t=0'ina karsilik gelsin")
    args = ap.parse_args()

    df = pd.read_csv(args.tracks)
    if "real_x" not in df.columns or "real_y" not in df.columns:
        sys.exit("HATA: 'real_x'/'real_y' sutunlari yok - once apply_homography.py calistirin.")

    # Her arac icin ilk/son gorulme zamanini onceden hesapla (hizli erisim icin)
    track_spans = df.groupby("track_id")["time_s"].agg(["min", "max"])

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"HATA: video acilamadi: {args.video}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if args.time_offset > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.time_offset * 1000)

    sumo_cmd = [args.sumo_binary, "-n", args.net, "--start", "--quit-on-end",
                "--step-length", "0.1", "--no-step-log", "true"]
    if args.poly:
        sumo_cmd += ["-a", args.poly]
    if args.gui_settings and args.sumo_binary == "sumo-gui":
        sumo_cmd += ["--gui-settings-file", args.gui_settings]

    print(">>> SUMO baslatiliyor (araclar dogrudan video verisiyle kontrol edilecek)...")
    traci.start(sumo_cmd)

    # Bos bir "yer tutucu" rota tanimliyoruz - moveToXY ile keepRoute=2
    # kullanildiginda gercek rota onemsiz, sadece bir referans edge lazim.
    # ONEMLI: EN UZUN edge'i seciyoruz - kisa bir edge secilirse arac
    # saniyeler icinde rotayi "bitirip" (arrival) SUMO'dan kalici olarak
    # dusuyor ve o ID bir daha ASLA yeniden eklenemiyor (SUMO kurali).
    all_edges = traci.edge.getIDList()
    placeholder_edge = None
    best_length = -1
    for e in all_edges:
        if e.startswith(":"):  # ic (internal) kenarlari atla
            continue
        try:
            length = traci.lane.getLength(f"{e}_0")
        except traci.exceptions.TraCIException:
            continue
        if length > best_length:
            best_length = length
            placeholder_edge = e
    print(f">>> Yer tutucu edge: {placeholder_edge} (uzunluk={best_length:.0f}m)")
    traci.route.add("dummy_route", [placeholder_edge])

    # Varsayilan goruntu TUM agi gosterecek sekilde uzaklasmis olur -
    # kavsagin gercek koordinatlarina otomatik yakinlasiyoruz.
    # ONEMLI: view ID'yi VARSAYMAK yerine SUMO'dan soruyoruz (bazen
    # "View #0" degil farkli bir isim donebiliyor).
    try:
        view_ids = traci.gui.getIDList()
        print(f">>> Mevcut GUI view'lari: {view_ids}")
        if view_ids:
            view_id = view_ids[0]
            traci.gui.setBoundary(view_id, 500, 700, 700, 900)
            print(f">>> Zoom ayarlandi: {view_id} -> (500,700)-(700,900)")
        else:
            print("[uyari] hicbir GUI view bulunamadi - zoom ayarlanamadi.")
    except traci.exceptions.TraCIException as e:
        print(f"[HATA] otomatik zoom ayarlanamadi: {e}")

    active_vehicles = set()
    step_length = 0.1
    sim_time = 0.0
    frame_idx = 0
    writer = None

    try:
        while True:
            ret, video_frame = cap.read()
            if not ret:
                print(">>> Video bitti.")
                break

            # Bu video karesinde "canli" olmasi gereken araclari, HENUZ ADIM
            # ATMADAN ONCE belirleyip ekleme/silme islemini yapiyoruz - boylece
            # traci.vehicle.add() ile eklenen aracin GERCEKTEN var olmasi icin
            # gereken simulationStep() cagrisi, ekleme SONRASINDA gerceklesir.
            target_time = frame_idx / video_fps
            current_ids = set(track_spans[
                (track_spans["min"] <= target_time) & (track_spans["max"] >= target_time)
            ].index)

            for tid in current_ids - active_vehicles:
                veh_id = f"v{tid}"
                try:
                    traci.vehicle.add(veh_id, "dummy_route", typeID="DEFAULT_VEHTYPE",
                                       departPos="0", departSpeed="0")
                    traci.vehicle.setSpeedMode(veh_id, 0)
                    active_vehicles.add(tid)
                except traci.exceptions.TraCIException as e:
                    if frame_idx < 20:
                        print(f"[HATA] arac eklenemedi v{tid}: {e}")

            for tid in active_vehicles - current_ids:
                veh_id = f"v{tid}"
                try:
                    traci.vehicle.remove(veh_id)
                except traci.exceptions.TraCIException:
                    pass
            active_vehicles &= current_ids  # sadece basariyla eklenmis + hala gecerli olanlari tut

            # SIMDI adimi ilerlet - bu, az once eklenen araclari GERCEKTEN
            # simulasyona sokar (add() sadece "planlar", step() "gerceklestirir")
            while sim_time < target_time:
                traci.simulationStep()
                sim_time = round(sim_time + step_length, 2)

            # HER aktif araci, o anki GERCEK konumuna yerlestir (kuklacilik).
            # "Yer tutucu" rota cok kisa oldugu icin arac bazen eklendikten
            # hemen sonra rotayi "bitirip" SUMO'dan dusebiliyor (arrival) -
            # boyle bir durumda TESPIT EDIP HEMEN YENIDEN EKLIYORUZ
            # (kendi kendini onaran mantik), hata vermek yerine.
            known_ids = set(traci.vehicle.getIDList())
            for tid in list(active_vehicles):
                veh_id = f"v{tid}"
                if veh_id not in known_ids:
                    try:
                        traci.vehicle.add(veh_id, "dummy_route", typeID="DEFAULT_VEHTYPE",
                                           departPos="0", departSpeed="0")
                        traci.vehicle.setSpeedMode(veh_id, 0)
                        traci.simulationStep()  # gercekten var olmasi icin bir adim daha
                    except traci.exceptions.TraCIException:
                        active_vehicles.discard(tid)
                        continue

                track_rows = df[df["track_id"] == tid]
                closest = track_rows.iloc[(track_rows["time_s"] - sim_time).abs().argmin()]
                try:
                    traci.vehicle.moveToXY(
                        veh_id, edgeID=placeholder_edge, laneIndex=0,
                        x=closest["real_x"], y=closest["real_y"],
                        angle=-1, keepRoute=2,
                    )
                except traci.exceptions.TraCIException as e:
                    if frame_idx < 10:
                        print(f"[HATA] moveToXY basarisiz v{tid}: {e}")

            # Ekran goruntusu al
            shot_path = os.path.abspath(f"_frame_tmp_{frame_idx:06d}.png")
            sim_frame = None
            if args.sumo_binary == "sumo-gui":
                try:
                    traci.gui.screenshot("View #0", shot_path)
                    ok = False
                    for _ in range(100):  # en fazla 2 saniye bekle
                        if os.path.exists(shot_path) and os.path.getsize(shot_path) > 0:
                            ok = True
                            break
                        time.sleep(0.02)
                    if ok:
                        sim_frame = cv2.imread(shot_path)
                        os.remove(shot_path)
                    elif frame_idx < 5:
                        print(f"[uyari] kare {frame_idx}: ekran goruntusu dosyasi olusmadi: {shot_path}")
                except traci.exceptions.TraCIException as e:
                    if frame_idx < 5:
                        print(f"[HATA] traci.gui.screenshot basarisiz (kare {frame_idx}): {e}")
            if sim_frame is None:
                sim_frame = np.zeros_like(video_frame)
                cv2.putText(sim_frame, "Screenshot alinamadi", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            video_r = resize_to_height(video_frame, args.panel_height)
            sim_r = resize_to_height(sim_frame, args.panel_height)
            cv2.putText(video_r, "GERCEK VIDEO", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(sim_r, "DIJITAL IKIZ (VIDEODAN BIREBIR)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            combined = cv2.hconcat([video_r, sim_r])
            combined = draw_info_bar(combined, sim_time, len(active_vehicles), combined.shape[1])

            if args.output:
                if writer is None:
                    h, w = combined.shape[:2]
                    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                                              video_fps, (w, h))
                writer.write(combined)

            if args.live:
                cv2.imshow("Gercek Video vs Dijital Ikiz (birebir replay)", combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print(">>> Kullanici tarafindan durduruldu.")
                    break

            frame_idx += 1

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.live:
            cv2.destroyAllWindows()
        traci.close()

    print(">>> Bitti.")
    if args.output:
        print(f">>> Kaydedildi: {args.output}")


if __name__ == "__main__":
    main()