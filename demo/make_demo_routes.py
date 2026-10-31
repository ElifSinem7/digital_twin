#!/usr/bin/env python3
"""
make_demo_routes.py
=====================
sumo/routes_replay.rou.xml'in SADECE DEMO icin bir kopyasini uretir
(orijinal dosya degismez) ve onu kullanan bir sumocfg yazar.

Neden: replay_vehicles.py her araci, videoda kavsaga girdigi saniyede
giris edge'inin BASINA ve departSpeed=0 ile ekliyor. Bati girisi
(428067756) 184 m uzunlugunda oldugu icin araclar kamera goruntusunun
disinda, duruştan kalkarak basliyor ve kavsaga videodan ~20-30 s gec
ulasiyor. Demo kopyasinda:
  - departPos: arac stop cizgisine en fazla --approach-m metre kala baslar
  - departSpeed="max": arac yol hizinda girer (durup kalkmaz)
  - departLane="best": dolu serit yuzunden ekleme gecikmez
  - maxSpeed alt siniri --min-speed'e yukseltilir (cok yavas olcumler icin)

Kullanim:
    python make_demo_routes.py --approach-m 40 --min-speed 8
"""
import argparse
import os
import re
import sys

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("HATA: SUMO_HOME tanimli degil.")

import sumolib  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SUMOCFG = """<?xml version="1.0" encoding="UTF-8"?>
<!-- OTOMATIK URETILDI - demo/make_demo_routes.py (sadece demo kaydi icin) -->
<configuration>
    <input>
        <net-file value="../network/bellevue_116th_ne12th.net.xml"/>
        <route-files value="routes_demo.rou.xml"/>
        <additional-files value="../network/bellevue_116th_ne12th.poly.xml"/>
        <gui-settings-file value="../sumo/view_settings.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="0.1"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
        <collision.action value="warn"/>
        <ignore-route-errors value="true"/>
    </processing>
</configuration>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", default=os.path.join(ROOT, "sumo", "routes_replay.rou.xml"))
    ap.add_argument("--net", default=os.path.join(ROOT, "network", "bellevue_116th_ne12th.net.xml"))
    ap.add_argument("--out-routes", default=os.path.join(HERE, "routes_demo.rou.xml"))
    ap.add_argument("--out-cfg", default=os.path.join(HERE, "digital_twin_demo.sumocfg"))
    ap.add_argument("--approach-m", type=float, default=40.0,
                    help="Arac stop cizgisine en fazla kac metre kala baslasin")
    ap.add_argument("--min-speed", type=float, default=8.0,
                    help="maxSpeed alt siniri (m/s); orijinalde 3.0")
    args = ap.parse_args()

    net = sumolib.net.readNet(args.net)
    edge_len = {}

    def fix_trip(m):
        trip = m.group(0)
        src = re.search(r'from="([^"]+)"', trip).group(1)
        if src not in edge_len:
            edge_len[src] = net.getEdge(src).getLength()
        pos = max(0.0, edge_len[src] - args.approach_m)
        trip = re.sub(r'\s(departSpeed|departPos|departLane)="[^"]*"', "", trip)
        trip = re.sub(r'maxSpeed="([0-9.]+)"',
                      lambda s: f'maxSpeed="{max(float(s.group(1)), args.min_speed):.1f}"', trip)
        extra = f' departPos="{pos:.1f}" departSpeed="max" departLane="best"'
        return trip[:-2].rstrip() + extra + "/>"

    with open(args.routes) as f:
        txt = f.read()
    out, n = re.subn(r"<trip [^>]*/>", fix_trip, txt)
    out = out.replace("<routes>", "<!-- DEMO KOPYASI: demo/make_demo_routes.py ile duzenlendi -->\n<routes>", 1)

    with open(args.out_routes, "w") as f:
        f.write(out)
    with open(args.out_cfg, "w") as f:
        f.write(SUMOCFG)
    print(f">>> {n} trip duzenlendi -> {args.out_routes}")
    for e, length in edge_len.items():
        print(f"    {e}: uzunluk {length:.1f} m -> departPos {max(0.0, length - args.approach_m):.1f}")
    print(f">>> sumocfg -> {args.out_cfg}")


if __name__ == "__main__":
    main()
