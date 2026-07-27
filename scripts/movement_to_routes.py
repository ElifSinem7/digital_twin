"""
Birden fazla saatlik videonun tracks.csv (vehicle_tracker.py ciktisi)
dosyalarini okur, config.json'daki giris/cikis bolgeleriyle (zones)
kesistirerek her aracin HANGI YAKLASIM KOLUNDAN GIRIP HANGI YAKLASIM
KOLUNDAN CIKTIGINI (duz/sag/sol hareketi) belirler, ve saat saat
degisen GERCEKCI bir routes.rou.xml uretir - yani gece 19:00'daki
trafik hacmi ile sabah 08:00'daki hacim SUMO'da da farkli olur, tipki
gercek videodaki gibi.

"""
import argparse
import glob
import json
import os

import pandas as pd
from shapely.geometry import Point, Polygon


def load_zones(config):
    entry_zones, exit_zones = {}, {}
    for name, pts in config["zones"].items():
        poly = Polygon(pts)
        if name.startswith("entry"):
            entry_zones[name] = poly
        elif name.startswith("exit"):
            exit_zones[name] = poly
    return entry_zones, exit_zones


def classify_track(rows, entry_zones, exit_zones):
    """rows: track_id'ye ait, zamana gore sıralı DataFrame satırları."""
    first_entry, last_exit = None, None
    for _, r in rows.iterrows():
        p = Point(r["cx"], r["cy"])
        if first_entry is None:
            for name, poly in entry_zones.items():
                if poly.contains(p):
                    first_entry = name
                    break
        for name, poly in exit_zones.items():
            if poly.contains(p):
                last_exit = name 
    return first_entry, last_exit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--zone-edge-map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video-duration-s", type=float, default=3600.0,
                     help="Her videonun gerçek süresi (varsayılan 3600s = 1 saat)")
    args = ap.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    with open(args.zone_edge_map) as f:
        zone_edge = json.load(f)

    entry_zones, exit_zones = load_zones(config)
    if not entry_zones or not exit_zones:
        raise SystemExit("HATA: config.json içinde entry_*/exit_* bölgesi bulunamadı.")

    track_files = sorted(glob.glob(os.path.join(args.tracks_dir, "*.csv")))
    if not track_files:
        raise SystemExit(f"HATA: {args.tracks_dir} içinde CSV bulunamadı.")

    flow_lines = []
    vtype_lines = [
        '    <vType id="car" vClass="passenger" accel="2.6" decel="4.5" sigma="0.5" '
        'length="4.8" minGap="2.5" maxSpeed="15.6" guiShape="passenger"/>',
        '    <vType id="bus" vClass="bus" accel="1.5" decel="4.0" sigma="0.3" '
        'length="12.0" minGap="3.0" maxSpeed="14.0" guiShape="bus"/>',
        '    <vType id="truck" vClass="truck" accel="1.3" decel="4.0" sigma="0.4" '
        'length="8.0" minGap="3.0" maxSpeed="14.0" guiShape="truck"/>',
        '    <vType id="motorcycle" vClass="motorcycle" accel="3.0" decel="5.0" sigma="0.5" '
        'length="2.2" minGap="2.0" maxSpeed="16.0" guiShape="motorcycle"/>',
    ]

    flow_id = 0
    for hour_idx, track_csv in enumerate(track_files):
        begin = hour_idx * args.video_duration_s
        end = begin + args.video_duration_s

        df = pd.read_csv(track_csv)
        if df.empty:
            print(f"[uyari] boş tracks dosyası, atlanıyor: {track_csv}")
            continue

        movement_counts = {}  # (entry, exit, cls) -> count
        for track_id, rows in df.sort_values("time_s").groupby("track_id"):
            entry, exit_ = classify_track(rows, entry_zones, exit_zones)
            if entry is None or exit_ is None:
                continue  # kavşağı geçmeden kaybolan / net olmayan izleri atla
            cls = rows["cls"].mode().iloc[0] if not rows["cls"].mode().empty else "car"
            key = (entry, exit_, cls)
            movement_counts[key] = movement_counts.get(key, 0) + 1

        n_matched = sum(movement_counts.values())
        n_total = df["track_id"].nunique()
        print(f"[{os.path.basename(track_csv)}] {n_matched}/{n_total} arac "
              f"giris-cikis ile eslesti -> sim {begin:.0f}-{end:.0f}s")

        for (entry, exit_, cls), count in movement_counts.items():
            if entry not in zone_edge or exit_ not in zone_edge:
                print(f"    [uyari] zone_to_edge.json'da eksik: {entry} / {exit_}, atlanıyor")
                continue
            vph = round(count * 3600.0 / args.video_duration_s, 1)
            if vph <= 0:
                continue
            flow_lines.append(
                f'    <flow id="f{flow_id}" type="{cls}" '
                f'from="{zone_edge[entry]}" to="{zone_edge[exit_]}" '
                f'begin="{begin:.0f}" end="{end:.0f}" vehsPerHour="{vph}"/>'
            )
            flow_id += 1

    with open(args.out, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write("<!-- OTOMATIK ÜRETİLDİ - movement_to_routes.py -->\n")
        f.write("<!-- Her <flow> bloğu, geröek videodan ölçülmüş, saat saat \n")
        f.write("     değişen gerçek araç hacmini temsil eder. -->\n")
        f.write("<routes>\n")
        f.write("\n".join(vtype_lines) + "\n")
        f.write("\n".join(flow_lines) + "\n")
        f.write("</routes>\n")

    print(f">>> {flow_id} flow üretildi -> {args.out}")


if __name__ == "__main__":
    main()