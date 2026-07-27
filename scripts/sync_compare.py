"""
Gercek trafik kamerasi videosu ile SUMO dijital ikizini YAN YANA,
SENKRON olarak calistirip karsilastiran arac.

- Solda: gercek video (dataset'ten)
- Sagda: SUMO-GUI simulasyon ekran goruntusu (arka planda gercek harita/
  uydu goruntusu decal olarak yuklu)
- Ust bilgi cubugu: simulasyon zamani + her yaklasim kolundaki anlik
  arac sayisi (SUMO tarafi icin TraCI'den canli okunur)
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

# SUMO_HOME/tools klasörünü PYTHONPATH'e ekleme (traci burada)
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("HATA: SUMO_HOME ortam degiskeni tanimli degil. "
              "Once SUMO kurulumunuzu ve SUMO_HOME'u ayarlayin.")

import traci  # noqa: E402


# Kavsaga giren gerçek edge ID'leri (netedit'te doğrulandı,
# cluster_53190763_5896114911_7619513267_7619513268 kavşağı).
APPROACH_EDGES = {
    "Kuzey (116th Ave NE)": "428067750#0",
    "Guney (116th Ave NE)": "-428067754#0",
    "Dogu (NE 12th St)": "428067759#0",
    "Bati (NE 12th St)": "428067756",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="Gercek kavşak videosunun yolu")
    p.add_argument("--sumocfg", required=True, help="SUMO .sumocfg dosyasının yolu")
    p.add_argument("--sumo-binary", default="sumo-gui",
                   help="'sumo-gui' (görsel) ya da 'sumo' (hızlı/görsel yok)")
    p.add_argument("--video-start-offset", type=float, default=0.0,
                   help="Videoda simulasyon t=0'ina karşılık gelen saniye")
    p.add_argument("--output", default=None,
                   help="Karşılaştırma videosunu bu dosyaya mp4 olarak kaydet")
    p.add_argument("--live", action="store_true",
                   help="Karşılaştırmayı ekranda canlı göster")
    p.add_argument("--panel-height", type=int, default=540,
                   help="Her iki panelin de ölçekleneceği yükseklik (px)")
    p.add_argument("--screenshot-dir", default="_frames",
                   help="SUMO ekran görüntülerinin geçici olarak yazıldığı klasör")
    return p.parse_args()


def get_vehicle_counts():
    """Her yaklaşım kolundaki anlık araç sayısını TraCI'den okur."""
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

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"HATA: Video acilamadi: {args.video}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Video başlangıç ofsetine atla
    if args.video_start_offset > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.video_start_offset * 1000)

    print(f">>> Video FPS: {video_fps:.2f}")
    print(">>> SUMO baslatiliyor...")

    sumo_cmd = [args.sumo_binary, "-c", args.sumocfg, "--start", "--quit-on-end"]
    traci.start(sumo_cmd)

    step_length = traci.simulation.getDeltaT()  # sumocfg'deki step-length
    # Video karesi başına kaç SUMO adımı atılacağını hesaplar, böylece simülasyon zamanı video zamanını takip eder.
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
                print(">>> Simülasyonda araç kalmadı, sonlandırılıyor.")
                break

            for _ in range(sim_steps_per_video_frame):
                traci.simulationStep()

            sim_time = traci.simulation.getTime()
            counts = get_vehicle_counts()

            # SUMO-GUI ekran görüntüsünü al
            shot_path = os.path.join(args.screenshot_dir, f"f{frame_idx:06d}.png")
            try:
                traci.gui.screenshot("View #0", shot_path)
                for _ in range(20):
                    if os.path.exists(shot_path) and os.path.getsize(shot_path) > 0:
                        break
                    time.sleep(0.01)
                sim_frame = cv2.imread(shot_path)
            except traci.exceptions.TraCIException:
                sim_frame = None

            if sim_frame is None:
                sim_frame = np.zeros_like(video_frame)
                cv2.putText(sim_frame, "SUMO görüntüsü alınamadı (sumo-gui gerekli)",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            video_r = resize_to_height(video_frame, args.panel_height)
            sim_r = resize_to_height(sim_frame, args.panel_height)

            cv2.putText(video_r, "GERÇEK VIDEO", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(sim_r, "DİJİTAL İKİZ (SUMO)", (10, 30),
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
                cv2.imshow("Gerçek Video vs Dijital İkiz - 116th Ave NE & NE 12th St",
                           combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print(">>> Kullanıcı tarafından durduruldu.")
                    break

            # geçici screenshot dosyasını temizle
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
        print(f">>> Karşılaştırma videosu kaydedildi: {args.output}")


if __name__ == "__main__":
    main()