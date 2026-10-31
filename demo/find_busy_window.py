#!/usr/bin/env python3
"""
find_busy_window.py
=====================
Simulasyonu GUI'siz (sumo) hizlica kosturup, kayit goruntusunun icindeki
(kavsak merkezi +-half_width) arac sayisini her saniye olcer ve
  - baslangic aninda ekranda en az --min-start arac olan,
  - pencere boyunca ekrandaki toplam arac-saniyesi en yuksek
--duration saniyelik araligin baslangicini yazdirir (stdout'a sadece sayi).

Kullanim:
    python find_busy_window.py --sumocfg digital_twin_demo.sumocfg --duration 90
"""
import argparse
import json
import os
import subprocess
import sys

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("HATA: SUMO_HOME tanimli degil.")

import traci  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sumocfg", required=True)
    ap.add_argument("--duration", type=float, default=90.0)
    ap.add_argument("--search-until", type=float, default=900.0,
                    help="Ilk kac saniye icinde aranacak")
    ap.add_argument("--zone-edge-map", default="../zone_to_edge.json")
    ap.add_argument("--half-width", type=float, default=100.0)
    ap.add_argument("--min-start", type=int, default=2)
    args = ap.parse_args()

    traci.start(["sumo", "-c", args.sumocfg, "--no-step-log", "true", "--no-warnings", "true"],
                stdout=subprocess.DEVNULL)
    with open(args.zone_edge_map) as f:
        zone_edge = json.load(f)
    ends = [traci.lane.getShape(f"{e}_0")[-1] for z, e in zone_edge.items() if z.startswith("entry")]
    cx = sum(p[0] for p in ends) / len(ends)
    cy = sum(p[1] for p in ends) / len(ends)
    hw, hh = args.half_width, args.half_width * 9 / 16

    counts = []  # her tam saniyedeki goruntu ici arac sayisi
    while traci.simulation.getTime() < args.search_until + args.duration:
        traci.simulationStep()
        t = traci.simulation.getTime()
        if abs(t - round(t)) < 1e-6:
            n = 0
            for v in traci.vehicle.getIDList():
                x, y = traci.vehicle.getPosition(v)
                if abs(x - cx) <= hw and abs(y - cy) <= hh:
                    n += 1
            counts.append(n)
    traci.close()

    dur = int(args.duration)
    best_start, best_score = 0, -1
    for s in range(0, min(int(args.search_until), len(counts) - dur)):
        if counts[s] < args.min_start:
            continue
        score = sum(counts[s:s + dur])
        if score > best_score:
            best_start, best_score = s, score
    print(f"[bilgi] secilen aralik {best_start}-{best_start + dur}s, baslangicta ekranda "
          f"{counts[best_start]} arac, ortalama {best_score / dur:.1f} arac", file=sys.stderr)
    print(best_start)


if __name__ == "__main__":
    main()
