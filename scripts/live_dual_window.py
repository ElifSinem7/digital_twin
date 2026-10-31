#!/usr/bin/env python3
"""
live_dual_window.py
=====================
sync_compare.py'den FARKLI olarak, SUMO'nun goruntusunu screenshot ile
"cekip" videoyla birlestirmeye CALISMAZ (bu ortamda kararsiz cikti).
Bunun yerine:
  1) sumo-gui'yi kendi ayri penceresinde, NORMAL sekilde acar (canli,
     gercek zamanli calisir - screenshot'a hic gerek yok).
  2) Videoyu AYRI bir OpenCV penceresinde, sumo-gui ile SENKRON
     ilerleyen bir hizda oynatir.

Siz iki pencereyi ekranda yan yana surukleyip yerlestirirsiniz -
sunumda "canli, iki panelli" bir gorunum elde edersiniz, ama ekran
goruntusu/birlestirme katmaninin kararsizligi olmadan.

Kurulum:
    pip install opencv-python

Kullanim:
    python live_dual_window.py \
        --video ../videos/Bellevue_116th_NE12th__2017-09-10_19-08-25.mp4 \
        --sumocfg ../sumo/digital_twin.sumocfg
"""
import argparse
import os
import sys
import threading
import time

import cv2

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("HATA: SUMO_HOME tanimli degil.")

import traci  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--sumocfg", required=True)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"HATA: video acilamadi: {args.video}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_delay = 1.0 / video_fps

    print(">>> SUMO-GUI baslatiliyor (kendi penceresinde, canli)...")
    sumo_cmd = ["sumo-gui", "-c", args.sumocfg, "--start"]
    traci.start(sumo_cmd)

    try:
        view_ids = traci.gui.getIDList()
        if view_ids:
            traci.gui.setBoundary(view_ids[0], 500, 700, 700, 900)
    except traci.exceptions.TraCIException:
        pass

    step_length = traci.simulation.getDeltaT()
    print(f">>> Simulasyon adim uzunlugu: {step_length}s, video kare suresi: {frame_delay:.4f}s")

    print(">>> Video penceresi aciliyor - iki pencereyi ekranda yan yana")
    print(">>> surukleyip yerlestirin, sonra sunuma baslayin.")
    print(">>> Video penceresinde 'q' ile durdurabilirsiniz.")

    cv2.namedWindow("GERCEK VIDEO", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("GERCEK VIDEO", 960, 540)

    frame_idx = 0
    next_frame_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print(">>> Video bitti.")
            break

        if traci.simulation.getMinExpectedNumber() <= 0:
            print(">>> Simulasyonda arac kalmadi.")
            break

        # ONEMLI DUZELTME: sabit "kare basina N adim" yerine, BIRIKEN
        # gercek zamana gore adim atiyoruz. Adim uzunlugu (0.1s) video kare
        # suresinden (~0.033s) BUYUK oldugu icin, "kare basina en az 1 adim"
        # mantigi simulasyonu videodan ~3 KAT HIZLI calistiriyordu - bu,
        # araclarin videoyla ayni anda gelmemesinin asil sebebiydi.
        target_sim_time = frame_idx * frame_delay
        while traci.simulation.getTime() < target_sim_time:
            traci.simulationStep()

        cv2.putText(frame, "GERCEK VIDEO", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow("GERCEK VIDEO", frame)

        # Gercek zamanla senkron kalmak icin: bir sonraki karenin tam
        # zamanini hesapla, o zamana kadar bekle (video cok hizli akmasin)
        next_frame_time += frame_delay
        sleep_time = next_frame_time - time.time()
        wait_ms = max(1, int(sleep_time * 1000))
        if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
            print(">>> Durduruldu.")
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    traci.close()
    print(">>> Bitti.")


if __name__ == "__main__":
    main()