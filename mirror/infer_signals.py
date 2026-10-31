#!/usr/bin/env python3
"""
infer_signals.py
==================
Okunabilir kafalardan (signal_reader.py: W ve S yaklasimlari) gelen durumlari, okunamayan
yaklasimlar (E, N) icin arac hareketinden cikarilan durumlarla birlestirir ve her 0.1 s
icin SUMO'ya verilecek setRedYellowGreenState dizisini uretir.

Hareketten cikarim (dur-kalk davranisi):
  - "gecis" olayi: yaklasimdan gelen arac kavsak kutusuna >= 2 m/s ile girer.
    Saga donenler sayilmaz (ABD'de kirmizida saga donus serbest).
  - Birbirine 6 s'den yakin gecisler tek bir yesil araliginda birlestirilir;
    aralik [ilk gecis - 2 s, son gecis + 1 s], sonuna 3.5 s sari eklenir.
  - Catisma kurali: E duz gecisi S duz yesilken olamaz; N, W duz yesilken olamaz.
  - Kanit yoksa: N icin S ile esanli (karsi yon) varsayimi, E icin W ile esanli varsayimi.
Yontemin guvenilirligi, ayni algoritma W ve S'ye uygulanip lamba okumasiyla karsilastirilarak olculur.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

if "SUMO_HOME" not in os.environ:
    sys.exit("HATA: SUMO_HOME tanimli degil.")
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import sumolib  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TLS = "cluster_53190763_5896114911_7619513267_7619513268_#3more"
RIGHT_OF = {"W": "S", "E": "N", "N": "W", "S": "E"}   # gelinen kol -> saga donunce gidilen kol
STEP = 0.1


def crossings(tracks):
    """Her arac icin kavsak kutusuna ilk giris ani, kokeni, hedefi, hizi."""
    out = []
    for tid, t in tracks.groupby("track_id"):
        t = t.sort_values("step")
        inj = t.in_junction.to_numpy()
        idx = np.flatnonzero(inj[1:] & ~inj[:-1]) + 1
        if len(idx) == 0:
            continue
        r = t.iloc[idx[0]]
        out.append(dict(track_id=tid, time_s=r.time_s, origin=r.origin, dest=r.dest, speed=r.speed))
    return pd.DataFrame(out)


def green_from_events(times, axis, merge=6.0, pre=2.0, post=1.0, yellow=3.5, forbid=None):
    """Olay zamanlarindan G/y/r dizisi (axis: 0.1 s'lik zaman ekseni)."""
    st = np.full(len(axis), "r", dtype="<U1")
    times = np.sort(np.asarray(times, float))
    if forbid is not None:  # catisan yesil sirasindaki olaylar donus sayilir, atilir
        ok = [not forbid[np.argmin(np.abs(axis - t))] for t in times]
        times = times[np.array(ok, bool)] if len(times) else times
    if len(times) == 0:
        return st, 0
    groups = [[times[0]]]
    for t in times[1:]:
        if t - groups[-1][-1] <= merge:
            groups[-1].append(t)
        else:
            groups.append([t])
    for g in groups:
        a, b = g[0] - pre, g[-1] + post
        st[(axis >= a) & (axis < b)] = "G"
        st[(axis >= b) & (axis < b + yellow) & (st != "G")] = "y"
    if forbid is not None:
        st[forbid & (st != "r")] = "r"
    return st, len(groups)


def to_char(s):
    return {"G": "G", "Y": "y", "y": "y", "R": "r", "r": "r", "F": "g", "-": "r"}.get(s, "r")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default=os.path.join(HERE, "mirror_tracks_384_504.csv"))
    ap.add_argument("--lamps", default=os.path.join(HERE, "signal_timeline_384_504.csv"))
    ap.add_argument("--net", default=os.path.join(ROOT, "network", "bellevue_116th_ne12th.net.xml"))
    ap.add_argument("--out", default=os.path.join(HERE, "signal_plan_384_504.csv"))
    args = ap.parse_args()

    tr = pd.read_csv(args.tracks)
    lamps = pd.read_csv(args.lamps).set_index("time_s")
    t0, t1 = np.ceil(lamps.index.min() * 10) / 10, np.floor(lamps.index.max() * 10) / 10
    axis = np.round(np.arange(t0, t1 + 1e-9, STEP), 1)
    L = lamps.reindex(lamps.index.union(axis)).ffill().loc[axis]

    cr = crossings(tr)
    cr = cr[(cr.speed >= 2.0) & (cr.origin.isin(list("NESW")))]
    nonright = cr[cr.dest != cr.origin.map(RIGHT_OF)]
    print(f">>> {len(cr)} kavsaga giris olayi (>=2 m/s); saga donusler haric {len(nonright)}")
    print("    kokene gore:", nonright.origin.value_counts().to_dict())

    # --- yontem dogrulamasi: W ve S icin hareketten cikarim vs lamba ---
    print(">>> Yontem dogrulamasi (hareketten cikarim vs lamba okumasi, 0.1 s'lik tum adimlar):")
    for arm in "WS":
        lamp = np.array([{"G": "G", "Y": "y", "R": "r"}.get(v, "r") for v in L[f"{arm}_thru"]])
        inf, ng = green_from_events(nonright[nonright.origin == arm].time_s, axis)
        gl = lamp == "G"; gi = inf == "G"
        print(f"    {arm}: uyum {np.mean((lamp == inf) | ((lamp != 'G') & (inf != 'G'))):.1%} (yesil/yesil-degil), "
              f"lamba yesilinin {np.mean(gi[gl]):.0%}'i bulundu, cikarilan yesilin {np.mean(gl[gi]) if gi.any() else 0:.0%}'i dogru, "
              f"{ng} yesil araligi")

    # --- E ve N ---
    # 1) Kural (birincil): karsi yonun sola donusu YANIP SONEN SARI OK (FYA) gosteriyorsa,
    #    bu yonun duz akisi yesildir (izinli sola donus karsi duz akisa yol verir). Ok sabit
    #    sariya donerse karsi duz akis da sariya doner.
    def from_fya(left):
        st = np.full(len(left), "r", dtype="<U1")
        st[left == "F"] = "G"
        prev = ""   # Y'den onceki son farkli durum
        for i in range(len(left)):
            if left[i] == "Y" and prev == "F":
                st[i] = "y"
            if left[i] != "Y":
                prev = left[i]
        return st
    E_rule = from_fya(L["W_left"].to_numpy())
    N_rule = from_fya(L["S_left"].to_numpy())
    # 2) Hareket (destek): dur-kalk gecis olaylari
    s_green = (L["S_thru"] == "G").to_numpy()
    w_green = (L["W_thru"] == "G").to_numpy()
    E_mv, nE = green_from_events(nonright[nonright.origin == "E"].time_s, axis, forbid=s_green)
    N_mv, nN = green_from_events(nonright[nonright.origin == "N"].time_s, axis, forbid=w_green | (E_rule == "G"))
    for nm, rule, ev in (("E", E_rule, nonright[nonright.origin == "E"].time_s),
                         ("N", N_rule, nonright[nonright.origin == "N"].time_s)):
        inside = [rule[np.argmin(np.abs(axis - t))] in "Gy" for t in ev]
        outside = [f"{t:.1f}s" for t, ok in zip(ev, inside) if not ok]
        print(f">>> {nm}: FYA kuralinin yesil/sari araliklarina dusen hareket olayi: {sum(inside)}/{len(inside)}"
              + (f"  (disarida: {', '.join(outside)})" if outside else ""))
        if outside:
            o = nonright[(nonright.origin == nm) & nonright.time_s.round(1).isin([float(x[:-1]) for x in outside])]
            print("      ", o[["track_id", "time_s", "origin", "dest", "speed"]].to_dict("records"))
    E = np.where((E_rule == "r") & (E_mv == "G") & ~s_green, "G", E_rule)
    N = np.where((N_rule == "r") & (N_mv == "G") & ~w_green, "G", N_rule)
    src_E = np.where(E_rule != "r", "FYA kurali", np.where(E != "r", "hareket", "kirmizi"))
    src_N = np.where(N_rule != "r", "FYA kurali", np.where(N != "r", "hareket", "kirmizi"))
    print(f">>> Hareketin kurala ekledigi yesil: E {np.sum(src_E == 'hareket') * STEP:.1f} s, "
          f"N {np.sum(src_N == 'hareket') * STEP:.1f} s")
    for nm, arr in (("E", E), ("N", N)):
        seg = np.flatnonzero(np.r_[True, arr[1:] != arr[:-1]])
        print(f"    {nm}: " + " ".join(f"{arr[i]}@{axis[i]:.1f}" for i in seg))
    # S/W durumlarina gore E/N'nin catismasiz oldugunu kontrol et
    print(f"    catisma: E yesil & S yesil = {np.sum((E == 'G') & s_green)} adim; "
          f"N yesil & W yesil = {np.sum((N == 'G') & w_green)} adim")

    # --- SUMO durum dizisi ---
    net = sumolib.net.readNet(args.net, withPedestrianConnections=True)
    tls = net.getTLS(TLS)
    node = net.getNode(TLS)
    cx, cy = node.getCoord()[:2]
    links = {}
    for inL, outL, idx in tls.getConnections():
        e = inL.getEdge()
        x, y = inL.getShape()[0]
        if np.hypot(x - cx, y - cy) < 1.0:  # cok kisa kenar: bir onceki kenardan yon al
            x, y = e.getFromNode().getCoord()[:2]
        ang = np.degrees(np.arctan2(y - cy, x - cx))
        arm = "E" if -45 <= ang < 45 else "N" if 45 <= ang < 135 else "S" if -135 <= ang < -45 else "W"
        conn = [c for c in inL.getOutgoing() if c.getToLane() == outL]
        d = conn[0].getDirection() if conn else "s"
        ped = not inL.allows("passenger")
        links[idx] = (arm, d, ped)
    n_links = max(links) + 1
    print(">>> TLS baglantilari:", " ".join(f"{i}:{a}{d}{'(yaya)' if p else ''}" for i, (a, d, p) in sorted(links.items())))
    lane_state = {"W": (L["W_thru"].to_numpy(), L["W_left"].to_numpy()),
                  "S": (L["S_thru"].to_numpy(), L["S_left"].to_numpy()),
                  "E": (E, None), "N": (N, None)}
    states = []
    for k in range(len(axis)):
        s = []
        for i in range(n_links):
            if i not in links or links[i][2]:
                s.append("r"); continue
            arm, d, _ = links[i]
            thru, left = lane_state[arm]
            tc = to_char(thru[k])
            if d in ("l", "L", "t"):
                if left is not None:
                    s.append(to_char(left[k]))
                else:
                    s.append("g" if tc == "G" else tc)
            else:
                s.append(tc)
        states.append("".join(s))
    out = pd.DataFrame({"time_s": axis, "W_thru": L["W_thru"].to_numpy(), "W_left": L["W_left"].to_numpy(),
                        "S_thru": L["S_thru"].to_numpy(), "S_left": L["S_left"].to_numpy(),
                        "E_thru": E, "E_src": src_E, "N_thru": N, "N_src": src_N, "state": states})
    out.to_csv(args.out, index=False)
    print(f">>> {len(out)} adim -> {args.out}  (ornek durum @{axis[0]}: {states[0]})")


if __name__ == "__main__":
    main()
