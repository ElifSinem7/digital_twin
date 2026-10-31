#!/usr/bin/env python3
"""
replay_vehicles.py
====================
movement_to_routes.py'den FARKLI olarak, videodaki her GERCEK araci
istatistiksel bir akisa (flow) donusturmek yerine, TEK TEK, gercek
goruldugu saniyede simulasyona sokan bir routes dosyasi uretir.

Boylece sync_compare.py ile video ve simulasyonu yan yana koydugunuzda,
videoda 14. saniyede kavsaga giren araba, simulasyonda da (yaklasik)
14. saniyede kavsaga girer - "istatistiksel benzer trafik" degil,
"aynı arac akisinin tekrar oynatilmasi" oluyor.

Not: Aracin varis zamani (depart) VIDEODAN birebir alinir. Ama aracin
kavsak icindeki HIZI/DAVRANISI hala SUMO'nun kendi fizik motoruna
(car-following model) birakilir - yani "ayni ana kadar ayni yerde"
degil, "ayni zamanda kavsaga giren, sonra SUMO kurallariyla hareket
eden" bir arac. Piksel-piksel birebir ayni hareketi istiyorsaniz (tam
"kayit tekrari"), bu TraCI ile her adimda aracin konumunu elle
zorlamayi (moveToXY) gerektirir - bu, "simulasyon" olmaktan cikip
"video oynatici" olur, cok daha kirilgan ve zaman alici bir is oldugu
icin burada YAPMIYORUZ; SUMO'nun kendi hareket mantigina birakiyoruz.

Kurulum:
    pip install pandas shapely

Kullanim:
    python replay_vehicles.py \
        --tracks ../tracks/2017-09-10_19_real.csv \
        --config ../config.json \
        --zone-edge-map ../zone_to_edge.json \
        --out ../sumo/routes_replay.rou.xml \
        --speeds ../speeds_2017-09-10_19.csv
"""
import argparse
import json

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
    first_entry, first_entry_time = None, None
    last_exit = None
    for _, r in rows.iterrows():
        p = Point(r["cx"], r["cy"])
        if first_entry is None:
            for name, poly in entry_zones.items():
                if poly.contains(p):
                    first_entry = name
                    first_entry_time = r["time_s"]
                    break
        for name, poly in exit_zones.items():
            if poly.contains(p):
                last_exit = name
    return first_entry, first_entry_time, last_exit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True,
                     help="Tek bir videonun tracks.csv'si (istege bagli: real_x/real_y icerebilir)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--zone-edge-map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--speeds", default=None,
                     help="speed_extractor.py ciktisi (istege bagli) - "
                          "verilirse her aracin maxSpeed'i kendi olculen hizina yakin ayarlanir")
    args = ap.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    with open(args.zone_edge_map) as f:
        zone_edge = json.load(f)

    entry_zones, exit_zones = load_zones(config)
    df = pd.read_csv(args.tracks)
    if df.empty:
        raise SystemExit("HATA: tracks dosyasi bos.")

    speeds_lookup = {}
    if args.speeds:
        sdf = pd.read_csv(args.speeds)
        # ONEMLI: medyan (avg_speed_mps) DEGIL, p90 kullaniyoruz - medyan,
        # aracin kirmizi isikta bekledigi (duruk) sureyi de icerdigi icin
        # gercek seyir hizini degil, "cogunlukla duruyordu" bilgisini
        # yansitiyordu (bazi araclarda maxSpeed=0.0-0.1 gibi anlamsiz
        # degerler cikiyordu). p90, aracin GERCEKTEN hareket ettigi
        # anlardaki hizina cok daha yakin bir tahmin veriyor.
        speeds_lookup = dict(zip(sdf["track_id"], sdf["p90_speed_mps"]))

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

    vehicles = []  # (depart_time, xml_satiri) - sona depart'a gore siralayacagiz
    n_matched, n_total = 0, 0

    for track_id, rows in df.sort_values("time_s").groupby("track_id"):
        n_total += 1
        entry, depart_time, exit_ = classify_track(rows, entry_zones, exit_zones)
        if entry is None or exit_ is None:
            continue
        if entry not in zone_edge or exit_ not in zone_edge:
            continue
        n_matched += 1

        cls = rows["cls"].mode().iloc[0] if not rows["cls"].mode().empty else "car"
        extra = ""
        if track_id in speeds_lookup:
            # bu aracin GERCEKTEN olculen hizini maxSpeed olarak veriyoruz
            # (ust sinir gibi davranir, SUMO yine de sinyal/trafige gore yavaslatir)
            # Cok dusuk (anlamsiz - olcum/tespit hatasi) degerlere karsi
            # makul bir alt sinir koyuyoruz (3 m/s ~ 11 km/s - yayaya
            # yakin bir hiz, gercek bir aracin altina dusmemesi gerekir)
            measured = max(speeds_lookup[track_id], 3.0)
            extra = f' departSpeed="0" maxSpeed="{measured:.1f}"'

        # ONEMLI: <vehicle> degil <trip> kullaniyoruz - SUMO sadece <trip> ve
        # <flow> icin from/to ile OTOMATIK rota hesaplamayi destekliyor.
        # <vehicle> gomulu bir <route> referansi gerektirir.
        line = (f'    <trip id="v{track_id}" type="{cls}" depart="{depart_time:.1f}" '
                f'from="{zone_edge[entry]}" to="{zone_edge[exit_]}"{extra}/>')
        vehicles.append((depart_time, line))

    if not vehicles:
        raise SystemExit("HATA: hicbir arac giris-cikis ile eslesmedi - config.json'daki "
                          "bolgeleri kontrol edin.")

    # SUMO, arac listesinin depart zamanina gore SIRALI olmasini bekliyor
    vehicles.sort(key=lambda v: v[0])

    with open(args.out, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write("<!-- OTOMATIK URETILDI - replay_vehicles.py -->\n")
        f.write("<!-- Videodaki HER GERCEK arac, gercek goruldugu saniyede -->\n")
        f.write("<!-- (depart=) tek tek tanimlanmistir - istatistiksel flow degil. -->\n")
        f.write("<routes>\n")
        f.write("\n".join(vtype_lines) + "\n")
        for _, line in vehicles:
            f.write(line + "\n")
        f.write("</routes>\n")

    print(f">>> {n_matched}/{n_total} gercek arac eslesip yazildi -> {args.out}")
    print(f">>> Ilk arac depart={vehicles[0][0]:.1f}s, son arac depart={vehicles[-1][0]:.1f}s")


if __name__ == "__main__":
    main()