#!/usr/bin/env python3
"""
compare_counts.py
===================
Ayna demosu icin karsilastirma tablolari (demo/bellevue_v2_counts.md):
  1) Yaklasim basina kavsaga giren arac sayisi (dakika dilimleri): VIDEO vs SUMO ayna vs ESKI demo
     - VIDEO: goruntu uzayinda (esleme KULLANILMADAN) kutu merkezlerinin kol poligonlarindan
       kavsak kutusuna gecisi; kol, kavsaga dogru hareket edilen ilk kol.
     - SUMO ayna: kayit sirasinda SUMO'daki konumlar (mirror_run_log.csv); SUMO kavsak sekline
       ilk giris ani, kol = giristen onceki son konumun yonu.
     - ESKI demo: demo/digital_twin_demo.sumocfg GUI'siz kosturulur (giris kenarinin yonu).
  2) t = 421 / 441 / 461 s'de bekleyen (duran) arac sayisi, yaklasim basina.
  3) Isik uyumu: her 3 s'de SUMO'daki duz akis rengi vs videodan okunan lamba (W, S).
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from matplotlib.path import Path

from fisheye_map import FisheyeGroundMap
from view_utils import JUNCTION_C

if "SUMO_HOME" not in os.environ:
    sys.exit("HATA: SUMO_HOME tanimli degil.")
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import sumolib  # noqa: E402
import traci  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
JUNCTION = "cluster_53190763_5896114911_7619513267_7619513268_#3more"
CENTRE_BOX = Path(np.array([(458, 222), (752, 163), (1045, 352), (642, 668), (512, 672)]))
C_IMG = np.array([750.0, 410.0])
LEGS = {"N": ["entry_north", "exit_north", "queue_north"], "E": ["entry_east", "exit_east", "queue_east"],
        "S": ["entry_south", "exit_south", "queue_south"], "W": ["entry_west", "exit_west", "queue_west"]}
# kavsaga GIREN trafigin aktigi poligonlar (duzeltilmis config.json adlariyla)
APPROACH = {"N": ["entry_north", "queue_north"], "E": ["entry_east", "queue_east"],
            "S": ["entry_south", "exit_south", "queue_south"], "W": ["entry_west", "queue_west"]}
OLD_GEO = {"428067750#0": "S", "-428067754#0": "N", "428067759#0": "E", "428067756": "W"}


def arm_of(x, y):
    a = math.degrees(math.atan2(y - JUNCTION_C[1], x - JUNCTION_C[0]))
    return "E" if -45 <= a < 45 else "N" if 45 <= a < 135 else "S" if -135 <= a < -45 else "W"


def video_side(boxes, ids, cfg, m, waits, max_dist):
    Z = {k: Path(np.array(v)) for k, v in cfg["zones"].items()}
    b = boxes.copy()
    b["root"] = [ids.get(int(t), {"root": int(t)})["root"] for t in b.track_id]
    b["status"] = [ids.get(int(t), {"status": "short"})["status"] for t in b.track_id]
    b = b[b.status != "static"]
    b["cx"] = (b.x1 + b.x2) / 2; b["cy"] = (b.y1 + b.y2) / 2
    b = b.sort_values(["root", "frame"])
    g = b.groupby("root")
    b["vx"] = (g.cx.shift(-15) - g.cx.shift(15)).fillna(0); b["vy"] = (g.cy.shift(-15) - g.cy.shift(15)).fillna(0)
    P = b[["cx", "cy"]].to_numpy(); V = b[["vx", "vy"]].to_numpy(); R = C_IMG - P
    sp = np.linalg.norm(V, axis=1)  # piksel / s (+-0.5 s fark)
    b["spd"] = sp
    b["radial"] = np.where(sp > 8, (V * R).sum(1) / (np.linalg.norm(R, axis=1) * np.maximum(sp, 1e-6)), 0)
    b["inbox"] = CENTRE_BOX.contains_points(P)
    for l, zs in LEGS.items():
        b[f"leg_{l}"] = np.any([Z[z].contains_points(P) for z in zs], axis=0)
    for l, zs in APPROACH.items():
        b[f"ap_{l}"] = np.any([Z[z].contains_points(P) for z in zs], axis=0)
    w = m.img_to_world(b[["bx", "by"]].to_numpy())
    b["in_valid"] = np.hypot(w[:, 0] - JUNCTION_C[0], w[:, 1] - JUNCTION_C[1]) <= max_dist
    ent = []
    for root, t in b.groupby("root"):
        c = [(t[t[f"leg_{l}"] & (t.radial > 0.3)].time_s.min(), l) for l in "NESW"]
        c = [x for x in c if not np.isnan(x[0])]
        if not c:
            continue
        t0, o = min(c)
        a = t[(t.time_s >= t0) & t.inbox]
        if len(a):
            ent.append((root, o, a.time_s.min()))
    ent = pd.DataFrame(ent, columns=["id", "origin", "t"])
    q = {}
    for T in waits:
        near = b[(b.time_s - T).abs() < 0.02]
        near = near.drop_duplicates("root")
        q[T] = {l: (int((near[f"ap_{l}"] & (near.spd < 10)).sum()),
                    int((near[f"ap_{l}"] & (near.spd < 10) & near.in_valid).sum())) for l in "NESW"}
    return ent, q


def sumo_mirror_side(log, net, waits):
    jp = Path(np.array(net.getNode(JUNCTION).getShape()))
    log = log.sort_values(["track_id", "step"])
    log["inj"] = jp.contains_points(log[["x", "y"]].to_numpy())
    ent, first_inside = [], 0
    for tid, t in log.groupby("track_id"):
        inj = t.inj.to_numpy()
        if inj[0]:
            first_inside += 1
        idx = np.flatnonzero(inj[1:] & ~inj[:-1]) + 1
        if len(idx):
            r0 = t.iloc[idx[0] - 1]
            ent.append((tid, arm_of(r0.x, r0.y), t.iloc[idx[0]].time_s))
    ent = pd.DataFrame(ent, columns=["id", "origin", "t"])
    g = log.groupby("track_id")
    log["d1"] = np.hypot(g.x.shift(-5) - g.x.shift(5), g.y.shift(-5) - g.y.shift(5))
    q = {}
    for T in waits:
        near = log[(log.time_s - T).abs() < 0.05].drop_duplicates("track_id")
        st = near[(near.d1 < 1.0) & ~near.inj]
        q[T] = {l: int(sum(arm_of(x, y) == l for x, y in zip(st.x, st.y))) for l in "NESW"}
    return ent, q, first_inside


def old_demo_side(t_end, waits):
    os.chdir(os.path.join(ROOT, "demo"))
    traci.start(["sumo", "-c", "digital_twin_demo.sumocfg", "--no-step-log", "true", "--no-warnings", "true"],
                stdout=subprocess.DEVNULL)
    first, cross, q = {}, {}, {}
    while traci.simulation.getTime() < t_end:
        traci.simulationStep()
        t = round(traci.simulation.getTime(), 1)
        for v in traci.simulation.getDepartedIDList():
            first[v] = traci.vehicle.getRoute(v)[0]
        for v in traci.vehicle.getIDList():
            if v not in cross and traci.vehicle.getLaneID(v).startswith(":cluster_53190763"):
                cross[v] = t
        if t in waits:
            d = {l: 0 for l in "NESW"}
            for v in traci.vehicle.getIDList():
                if v not in cross and traci.vehicle.getSpeed(v) < 0.5 and first.get(v) in OLD_GEO:
                    d[OLD_GEO[first[v]]] += 1
            q[t] = d
    traci.close()
    ent = pd.DataFrame([(v, OLD_GEO.get(first[v], "?"), t) for v, t in cross.items()], columns=["id", "origin", "t"])
    return ent, q


def per_bin(ent, bins, labels):
    e = ent[(ent.t >= bins[0]) & (ent.t < bins[-1])]
    tab = pd.crosstab(pd.cut(e.t, bins, right=False, labels=labels), e.origin)
    return tab.reindex(index=labels, columns=list("NESW"), fill_value=0)


def episodes(times, gap=0.15):
    """Ardisik zaman damgalarini (araliklari) birlestir: [(t0, t1), ...]"""
    times = np.sort(np.asarray(times, float))
    if len(times) == 0:
        return []
    out, a, b = [], times[0], times[0]
    for t in times[1:]:
        if t - b > gap:
            out.append((a, b)); a = t
        b = t
    out.append((a, b))
    return out


def remaining_differences(args, ids, boxes, m):
    fl = pd.read_csv(os.path.join(HERE, "mirror_run_log_frames.csv"))
    tr = pd.read_csv(os.path.join(HERE, "mirror_tracks_384_504.csv"))
    L = ["## 4. Panellerin hala farkli oldugu anlar", ""]
    # (a) sayac tutarsizligi: video kutusu + kesikli kutu != SUMO
    bad = fl[fl.video_boxes + fl.video_ghost != fl.sumo]
    L.append(f"**(a) Sayac tutarsizligi** (video kutusu + kesikli kutu != SUMO): {len(bad)}/{len(fl)} kare"
             + ("." if len(bad) == 0 else ": " + ", ".join(f"{a:.1f}-{b:.1f} s" for a, b in episodes(bad.video_t))))
    # (b) tespit kaybi: SUMO araci surduruyor, videoda o an kutu yok (kesikli kutu)
    g = fl[fl.video_ghost > 0]
    eps = [(a, b) for a, b in episodes(g.video_t) if b - a >= 0.3]
    L += ["", f"**(b) Kisa tespit kayiplari** (videoda kutu yok, SUMO araci <= 1 s surduruyor; videoda kesikli kutu): "
          f"{len(g)}/{len(fl)} kare ({len(g) / len(fl):.1%}). 0.3 s'den uzun olanlar:"]
    L += [f"- {a:.1f}-{b:.1f} s" for a, b in eps] or ["- yok"]
    # (c) videoda hareket eden ama SUMO'da olmayan araclar (gri kutu, gecerli bolge icinde)
    b = boxes[(boxes.time_s >= args.start) & (boxes.time_s < args.end)].copy()
    b["status"] = [ids.get(int(t), {"status": "short"})["status"] for t in b.track_id]
    gb = b[b.status != "mirror"]
    if len(gb):
        w = m.img_to_world(gb[["bx", "by"]].to_numpy())
        gb = gb[np.hypot(w[:, 0] - JUNCTION_C[0], w[:, 1] - JUNCTION_C[1]) <= args.max_dist]
    L += ["", "**(c) Gecerli bolgede videoda gorunen ama SUMO'ya aktarilmayan tespitler** (gri kutu; cogu < 1 s'lik "
          "kisa parcalar):"]
    rows = []
    for tid, t in gb.groupby("track_id"):
        span = np.hypot(t.bx.max() - t.bx.min(), t.by.max() - t.by.min())
        rows.append((t.time_s.min(), t.time_s.max(), tid, ids.get(int(tid), {"status": "short"})["status"], span, len(t)))
    rows.sort()
    L += [f"- {a:.1f}-{e:.1f} s: iz {tid} ({st}, {n} kutu, goruntude {sp:.0f} px hareket)" for a, e, tid, st, sp, n in rows] \
        or ["- yok"]
    # (d) gorsel serit oturtma kaymasi buyuk anlar (> 3 m)
    big = tr[(tr.snap_state == "snapped") & (tr.lat_shift > 3.0) & (tr.time_s >= args.start) & (tr.time_s < args.end)]
    L += ["", "**(d) SUMO'da konumun videodan > 3 m yana kaydirildigi anlar** (serit oturtma; ag/esleme farki):"]
    for tid, t in big.groupby("track_id"):
        for a, e in episodes(t.time_s, gap=0.25):
            if e - a >= 0.5:
                L.append(f"- {a:.1f}-{e:.1f} s: arac {tid} ({t.arm.iloc[0]} kolu, en fazla {t.lat_shift.max():.1f} m)")
    if big.empty:
        L.append("- yok")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, default=399.0)
    ap.add_argument("--end", type=float, default=489.0)
    ap.add_argument("--max-dist", type=float, default=40.0)
    ap.add_argument("--out-md", default=os.path.join(ROOT, "demo", "bellevue_v3_counts.md"))
    ap.add_argument("--title", default="Bellevue v3 demo")
    args = ap.parse_args()
    waits = [421.0, 441.0, 461.0]
    cfg = json.load(open(os.path.join(ROOT, "config.json")))
    ids = {int(k): v for k, v in json.load(open(os.path.join(HERE, "mirror_track_ids.json"))).items()}
    m = FisheyeGroundMap.load(os.path.join(HERE, "fisheye_mapping.json"))
    net = sumolib.net.readNet(os.path.join(ROOT, "network", "bellevue_116th_ne12th.net.xml"))
    boxes = pd.read_csv(os.path.join(ROOT, "tracks", "window_384_504_boxes.csv"))
    log = pd.read_csv(os.path.join(HERE, "mirror_run_log.csv"))
    tls = pd.read_csv(os.path.join(HERE, "mirror_run_log_tls.csv"))

    v_ent, v_q = video_side(boxes, ids, cfg, m, waits, args.max_dist)
    s_ent, s_q, first_inside = sumo_mirror_side(log, net, waits)
    o_ent, o_q = old_demo_side(args.end + 1, waits)

    bins = [args.start, args.start + 60, args.end]
    fmt = lambda s: f"{int(s) // 60}:{int(s) % 60:02d}"
    labels = [f"{fmt(a)}-{fmt(b)} ({int(b - a)} s)" for a, b in zip(bins[:-1], bins[1:])]
    V, S, O = per_bin(v_ent, bins, labels), per_bin(s_ent, bins, labels), per_bin(o_ent, bins, labels)
    lines = [f"# {args.title} - karsilastirma tablolari", "",
             f"Demo araligi: video t = {args.start:.0f}-{args.end:.0f} s (19:15:04-19:16:34). "
             "N/E/S/W = aracin GELDIGI cografi yon.", "",
             "## 1. Kavsaga giren arac sayisi (yaklasim basina)", "",
             "| dilim | yaklasim | VIDEO | SUMO ayna | ESKI demo |", "|---|---|---|---|---|"]
    for lab in labels:
        for a in "NESW":
            lines.append(f"| {lab} | {a} | {V.loc[lab, a]} | {S.loc[lab, a]} | {O.loc[lab, a]} |")
        lines.append(f"| {lab} | **toplam** | **{V.loc[lab].sum()}** | **{S.loc[lab].sum()}** | **{O.loc[lab].sum()}** |")
    tot_abs = int(np.abs(V.to_numpy() - S.to_numpy()).sum()); tot_abs_o = int(np.abs(V.to_numpy() - O.to_numpy()).sum())
    lines += ["", f"Hucre basina mutlak fark toplami (video ile): SUMO ayna **{tot_abs}**, eski demo **{tot_abs_o}**.",
              f"SUMO'da kavsak icinde ilk kez gorunen (kolu belirlenemeyen) arac: {first_inside}.", "",
              "## 2. Bekleyen (duran) arac sayisi", "",
              "Video: kavsaga giren seritlerde duran kutular (goruntu uzayi; parantez icinde: 40 m gecerli bolge icindekiler). "
              "SUMO: kavsak sekli disinda, 1 s'de < 1 m hareket eden araclar.", "",
              "| t (s) | kaynak | N | E | S | W |", "|---|---|---|---|---|---|"]
    for T in waits:
        lines.append(f"| {T:.0f} | VIDEO | " + " | ".join(f"{v_q[T][a][0]} ({v_q[T][a][1]})" for a in "NESW") + " |")
        lines.append(f"| {T:.0f} | SUMO ayna | " + " | ".join(str(s_q[T][a]) for a in "NESW") + " |")
        lines.append(f"| {T:.0f} | ESKI demo | " + " | ".join(str(o_q.get(T, {}).get(a, '-')) for a in "NESW") + " |")
    # isik uyumu
    links_W, links_S = [6, 7], [1, 2]   # duz akis baglantilari (infer_signals.py ciktisindaki indeksler)
    def sumo_col(state, idx):
        c = state[idx[0]]
        return "G" if c in "Gg" else "Y" if c == "y" else "R"
    tls["sW"] = [sumo_col(s, links_W) for s in tls.sumo_state]
    tls["sS"] = [sumo_col(s, links_S) for s in tls.sumo_state]
    aW = (tls.sW == tls.video_W).mean(); aS = (tls.sS == tls.video_S).mean()
    lines += ["", "## 3. Isik uyumu (her 3 s'de bir ornek)", "",
              f"SUMO'daki duz akis rengi = videodan okunan lamba: bati yaklasimi **{aW:.0%}** ({len(tls)} ornek), "
              f"guney yaklasimi **{aS:.0%}**.", ""]
    lines += remaining_differences(args, ids, boxes, m)
    md = "\n".join(lines)
    with open(args.out_md, "w") as f:
        f.write(md + "\n")
    print(md)
    print(f"\n>>> {args.out_md}")


if __name__ == "__main__":
    main()
