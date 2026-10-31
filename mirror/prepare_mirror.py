#!/usr/bin/env python3
"""
prepare_mirror.py
===================
Kutulu tespitlerden (detect_window.py) ayna modu icin 0.1 s'lik arac durumlari uretir.

Adimlar
  1. Kare duzeyinde cift kutu temizligi: ayni karede IoU > 0.6 olan iki farkli izden kisa olan atilir
     (ByteTrack bazen ayni araca iki ID verir).
  2. Sabit izler (guvenlik agi; asil suzme detect_window.py --mask ile takipten once yapilir).
  3. Iz birlestirme (stitching): kopan izler birlestirilir. Yeni parca, eski parcanin hareket
     dogrultusunda ONUNDE baslamali ve ayni yone gitmeli (aksi halde baska bir aracin parcasi).
  4. Kutunun alt-orta noktasi balikgozu eslemesiyle SUMO zeminine izdusurulur (HAM konum),
     0.1 s'lik eksene oturtulur (<= 1 s bosluk doldurulur), medyan(3)+ortalama(5) ile yumusatilir.
  5. Gecerli bolge: kavsak merkezine <= --max-dist m (varsayilan 40 m).
  6. Yon ve koken/hedef HAREKETTEN cikarilir (bolge etiketi kullanilmaz).
  7. Serit KILITLEME (histerezisli, sadece gorsel): kolda arac ayni yone akan bir SUMO seridine
     kilitlenir; ancak HAM konum serit sinirini --lane-margin m (1.0) asarak --lane-hold s (1.0)
     boyunca disarida kalirsa serit degistirir. Kenar sonunda ardil serite gecis serit
     degisimi sayilmaz. 6 m icinde ayni yonlu serit yoksa ham konum kullanilir.
     Kavsak kutusu icinde ham konum. Kaydirma vektoru 1 s'lik kayan ortalama ile yumusatilir.

Cikti CSV: step,time_s,track_id,cls,origin,dest,origin_inferred,dest_inferred,raw_x,raw_y,x,y,angle,
           speed,arm,inbound,in_junction,snap_state,lane_id,lat_shift
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from matplotlib.path import Path

from fisheye_map import FisheyeGroundMap

if "SUMO_HOME" not in os.environ:
    sys.exit("HATA: SUMO_HOME tanimli degil.")
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import sumolib  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
JUNCTION = "cluster_53190763_5896114911_7619513267_7619513268_#3more"
STEP = 0.1
LANE_W_NOMINAL = 3.2


def arm_of(x, y, cx, cy):
    a = math.degrees(math.atan2(y - cy, x - cx))
    return "E" if -45 <= a < 45 else "N" if 45 <= a < 135 else "S" if -135 <= a < -45 else "W"


class Lanes:
    """Bolgedeki arac seritleri: en yakin nokta, yon, kenar disina tasma, ardil iliskisi."""

    def __init__(self, net, cx, cy, radius):
        self.shape, self.width, self.succ = {}, {}, {}
        segs = []
        for e in net.getEdges(withInternal=False):
            for ln in e.getLanes():
                if not ln.allows("passenger"):
                    continue
                sh = np.array(ln.getShape(), float)
                if np.min(np.hypot(sh[:, 0] - cx, sh[:, 1] - cy)) > radius + 20:
                    continue
                lid = ln.getID()
                self.shape[lid] = sh; self.width[lid] = ln.getWidth()
                self.succ[lid] = {c.getToLane().getID() for c in ln.getOutgoing()}
                n = len(sh) - 1
                for j, (a, b) in enumerate(zip(sh[:-1], sh[1:])):
                    if np.linalg.norm(b - a) > 1e-6:
                        segs.append((a, b, lid, j == 0, j == n - 1))
        self.A = np.array([s[0] for s in segs]); self.B = np.array([s[1] for s in segs])
        self.lane = np.array([s[2] for s in segs])
        self.first = np.array([s[3] for s in segs]); self.last = np.array([s[4] for s in segs])
        d = self.B - self.A
        self.len2 = (d ** 2).sum(1)
        self.dir = d / np.sqrt(self.len2)[:, None]

    def connected(self, a, b):
        return b in self.succ.get(a, ()) or a in self.succ.get(b, ())

    def query(self, p, heading, cos_min=0.7):
        """{lane: (yanal mesafe, izdusum noktasi, yon, kenar_sonunu_asti, kenar_basindan_once)}"""
        raw_t = ((p - self.A) * (self.B - self.A)).sum(1) / self.len2
        t = np.clip(raw_t, 0, 1)
        proj = self.A + t[:, None] * (self.B - self.A)
        dist = np.linalg.norm(proj - p, axis=1)
        ok = (self.dir @ heading) > cos_min
        out = {}
        for i in np.flatnonzero(ok):
            lid = self.lane[i]
            if lid not in out or dist[i] < out[lid][0]:
                beyond_end = bool(self.last[i] and raw_t[i] > 1 and (raw_t[i] - 1) * math.sqrt(self.len2[i]) > 0.5)
                before_start = bool(self.first[i] and raw_t[i] < 0 and -raw_t[i] * math.sqrt(self.len2[i]) > 0.5)
                out[lid] = (dist[i], proj[i], self.dir[i], beyond_end, before_start)
        return out

    def signed_offset(self, p, lid):
        """Seridin merkez cizgisine isaretli yanal uzaklik (sol +); uclarda dogru boyunca uzatilir."""
        sh = self.shape[lid]
        best = (1e9, 0.0)
        for j, (a, b) in enumerate(zip(sh[:-1], sh[1:])):
            d = b - a; L2 = d @ d
            if L2 < 1e-12:
                continue
            t = ((p - a) @ d) / L2
            if j > 0:
                t = max(t, 0.0)
            if j < len(sh) - 2:
                t = min(t, 1.0)
            q = a + t * d
            dist = np.linalg.norm(p - q)
            if dist < best[0]:
                cross = d[0] * (p - a)[1] - d[1] * (p - a)[0]
                best = (dist, math.copysign(dist, cross))
        return best[1]


def stitch(bx, max_gap=2.0, base_px=50.0, px_per_s=25.0):
    """Kopan izleri birlestirir. A bittikten <= max_gap s sonra baslayan B, A'nin son hizindan tahmin
    edilen yere yakinsa VE (A hareketliyse) B A'nin hareket yonunde onunde baslayip ayni yone gidiyorsa
    A'nin devamidir. Sinif uyumu aranmaz (YOLO car/truck arasinda degisebiliyor)."""
    info = {}
    for tid, t in bx.groupby("track_id"):
        t = t.sort_values("time_s")
        head, tail = t.head(10), t.tail(10)
        def vel(s):
            dt = s.time_s.iloc[-1] - s.time_s.iloc[0]
            return ((s[["bx", "by"]].iloc[-1] - s[["bx", "by"]].iloc[0]) / dt).to_numpy() if dt > 0 else np.zeros(2)
        info[tid] = (t.time_s.iloc[0], t[["bx", "by"]].iloc[0].to_numpy(), t.time_s.iloc[-1],
                     t[["bx", "by"]].iloc[-1].to_numpy(), vel(tail), vel(head))
    cands, n_rej = [], 0
    for a, (_, _, ae, ap_, av, _) in info.items():
        for b, (bs, bp, _, _, _, bv) in info.items():
            gap = bs - ae
            if a == b or gap <= 0 or gap > max_gap:
                continue
            dist = np.linalg.norm(ap_ + av * gap - bp)
            if dist > base_px + px_per_s * gap:
                continue
            sa, sb = np.linalg.norm(av), np.linalg.norm(bv)
            if sa > 20:
                if (bp - ap_) @ (av / sa) < -10:          # B, A'nin gerisinde basliyor
                    n_rej += 1; continue
                if sb > 20 and (av @ bv) / (sa * sb) < 0.3:  # B baska yone gidiyor
                    n_rej += 1; continue
            cands.append((dist + 20 * gap, a, b))
    nxt, used_a, used_b = {}, set(), set()
    for _, a, b in sorted(cands):
        if a in used_a or b in used_b:
            continue
        nxt[a] = b; used_a.add(a); used_b.add(b)
    prev = {b: a for a, b in nxt.items()}
    root = {}
    for tid in info:
        r = tid
        while r in prev:
            r = prev[r]
        root[tid] = r
    bx = bx.copy()
    bx["track_id"] = bx.track_id.map(root)
    return bx, len(nxt), root, n_rej


def dedupe(bx, thr=0.6):
    """Ayni karede IoU > thr olan farkli izlerden, toplam tespiti az olanin kutusunu at."""
    size = bx.groupby("track_id").size()
    drop = []
    for _, f in bx.groupby("frame"):
        if len(f) < 2:
            continue
        b = f[["x1", "y1", "x2", "y2"]].to_numpy(); idx = f.index.to_numpy(); tids = f.track_id.to_numpy()
        for i in range(len(f)):
            for j in range(i + 1, len(f)):
                ix = max(0, min(b[i, 2], b[j, 2]) - max(b[i, 0], b[j, 0]))
                iy = max(0, min(b[i, 3], b[j, 3]) - max(b[i, 1], b[j, 1]))
                inter = ix * iy
                u = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1]) + (b[j, 2] - b[j, 0]) * (b[j, 3] - b[j, 1]) - inter
                if inter / max(u, 1e-9) > thr:
                    drop.append(idx[j] if size[tids[i]] >= size[tids[j]] else idx[i])
    return bx.drop(index=sorted(set(drop))), len(set(drop))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes", default=os.path.join(ROOT, "tracks", "window_384_504_boxes.csv"))
    ap.add_argument("--mapping", default=os.path.join(HERE, "fisheye_mapping.json"))
    ap.add_argument("--net", default=os.path.join(ROOT, "network", "bellevue_116th_ne12th.net.xml"))
    ap.add_argument("--out", default=os.path.join(HERE, "mirror_tracks_384_504.csv"))
    ap.add_argument("--max-dist", type=float, default=40.0, help="Gecerli bolge: kavsak merkezine en fazla (m)")
    ap.add_argument("--snap-max", type=float, default=6.0)
    ap.add_argument("--lane-margin", type=float, default=1.0, help="Serit degisimi icin sinir asim payi (m)")
    ap.add_argument("--lane-hold", type=float, default=1.0, help="Serit degisimi icin asimin surmesi gereken sure (s)")
    ap.add_argument("--no-drift-correction", action="store_true",
                    help="Kol basina sistematik yanal kayma duzeltmesini kapat")
    ap.add_argument("--stats-out", default=os.path.join(HERE, "mirror_prepare_stats.json"))
    args = ap.parse_args()
    hold_steps = int(round(args.lane_hold / STEP))

    m = FisheyeGroundMap.load(args.mapping)
    net = sumolib.net.readNet(args.net)
    node = net.getNode(JUNCTION)
    jpath = Path(np.array(node.getShape()))
    cx, cy = node.getCoord()[:2]
    lanes = Lanes(net, cx, cy, args.max_dist)

    bx = pd.read_csv(args.boxes)
    n_raw = bx.track_id.nunique()
    bx, n_dup = dedupe(bx)
    print(f">>> Cift kutu temizligi: {n_dup} kutu atildi (ayni karede baska bir izle IoU > 0.6)")
    # sabit izler (guvenlik agi) birlestirmeden ONCE
    g = bx.groupby("track_id")
    span0 = np.hypot(g.bx.max() - g.bx.min(), g.by.max() - g.by.min())
    dur0 = g.time_s.max() - g.time_s.min()
    med = g[["bx", "by"]].median()
    mask_json = os.path.join(HERE, "static_objects.json")
    if os.path.exists(mask_json):
        # Maske varsa: guvenlik agi SADECE bilinen sabit nesnelerin yerinde calisir. Genel "hareketsiz iz"
        # kurali kullanilmaz, cunku kirmizida uzun sure bekleyen gercek araclari da siliyordu.
        objs = [o["box"] for o in json.load(open(mask_json))["objects"] if o["static"]]
        spots = np.array([((x1 + x2) / 2, y2) for x1, y1, x2, y2 in objs]).reshape(-1, 2)
        static_long = pd.Index([])
    else:
        static_long = span0[(span0 < 15) & (dur0 > 5)].index
        spots = med.loc[static_long].to_numpy()
    near_spot = pd.Series(False, index=med.index)
    if len(spots):
        dmin = np.min(np.linalg.norm(med.to_numpy()[:, None, :] - spots[None], axis=2), axis=1)
        near_spot = pd.Series(dmin < 20, index=med.index)
    static0 = span0[(span0 < 15) & (span0.index.isin(static_long) | near_spot)].index
    st_bx = bx[bx.track_id.isin(static0)]
    bx, n_st, root_map, n_rej = stitch(bx[~bx.track_id.isin(static0)])
    print(f">>> {n_raw} ham iz; sabit (guvenlik agi) {len(static0)}; birlestirme {n_st} "
          f"(yon/konum tutarsizligi yuzunden reddedilen aday {n_rej}) -> {bx.track_id.nunique()} hareketli iz")
    g = bx.groupby("track_id")
    dur = g.time_s.max() - g.time_s.min()
    short = dur[(dur < 1.0) | (g.size() < 10)].index
    bx = bx[~bx.track_id.isin(short)].copy()

    w = m.img_to_world(bx[["bx", "by"]].to_numpy())
    bx["wx"], bx["wy"] = w[:, 0], w[:, 1]
    bx["k"] = np.round(bx.time_s / STEP).astype(int)

    rows, lc_rows = [], []
    n_total_steps = n_removed = 0
    Cv = np.array([cx, cy])

    # ---------------- 1. gecis: kinematik ----------------
    kin = []
    for tid, t in bx.groupby("track_id"):
        cls = t["cls"].mode().iloc[0]
        s = t.groupby("k")[["wx", "wy"]].median()
        s = s.reindex(range(s.index.min(), s.index.max() + 1))
        s = s.interpolate(limit=10, limit_area="inside")
        s = s.rolling(3, center=True, min_periods=1).median().rolling(5, center=True, min_periods=1).mean()
        s = s.dropna()
        if len(s) < 5:
            continue
        d = np.hypot(s.wx - cx, s.wy - cy)
        n_total_steps += len(s)
        n_removed += int((d > args.max_dist).sum())
        s = s[d <= args.max_dist]
        if len(s) < 5:
            continue
        P = s[["wx", "wy"]].to_numpy(); K = s.index.to_numpy()
        idx = np.arange(len(P))
        i0 = np.maximum(idx - 5, 0); i1 = np.minimum(idx + 5, len(P) - 1)
        V = (P[i1] - P[i0]) / np.maximum((K[i1] - K[i0]) * STEP, STEP)[:, None]
        spd = np.linalg.norm(V, axis=1)
        mv = spd > 1.0
        head = np.full((len(P), 2), np.nan)
        head[mv] = V[mv] / spd[mv, None]
        head = pd.DataFrame(head).ffill().bfill().to_numpy()
        injunc = jpath.contains_points(P)
        never_moved = not mv.any()
        if never_moved:
            to_c = Cv - P
            head = to_c / np.maximum(np.linalg.norm(to_c, axis=1), 1e-6)[:, None]
        inbound = ((Cv - P) * V).sum(1) > 0
        if never_moved:
            inbound[:] = True
        arms = np.array([arm_of(x, y, cx, cy) for x, y in P])
        kin.append(dict(tid=tid, cls=cls, P=P, K=K, spd=spd, mv=mv, head=head, injunc=injunc,
                        never_moved=never_moved, inbound=inbound, arms=arms))

    def segments(k):
        seg_id = np.cumsum(np.r_[True, (k["injunc"][1:] != k["injunc"][:-1]) | (k["arms"][1:] != k["arms"][:-1])])
        for sid in np.unique(seg_id):
            ii = np.flatnonzero(seg_id == sid)
            if not k["injunc"][ii[0]]:
                yield ii

    # ---------------- 2. sistematik yanal kayma (kol + yon basina) ----------------
    # Ayni kol ve yondeki tum araclarin, sabit bir serite gore yanal ofseti yol boyunca ayni egimle
    # degisiyorsa bu surucu davranisi degil, esleme/ag geometrisi farkidir -> egim kaldirilir.
    slopes = {}
    for k in kin:
        for ii in segments(k):
            if len(ii) < 15 or not k["mv"][ii].any():
                continue
            jn = ii[-1] if k["inbound"][ii].mean() > 0.5 else ii[0]      # kavsaga en yakin uc
            q = lanes.query(k["P"][jn], k["head"][jn])
            if not q:
                continue
            ref = min(q, key=lambda l: q[l][0])
            Pi = k["P"][ii]
            along = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Pi, axis=0), axis=1))]
            if along[-1] < 8:
                continue
            off_ = np.array([lanes.signed_offset(p, ref) for p in Pi])
            key = (k["arms"][ii[0]], bool(k["inbound"][ii].mean() > 0.5))
            slopes.setdefault(key, []).append((np.polyfit(along, off_, 1)[0], k["P"][jn], q[ref][2]))
    drift = {}
    for key, v in slopes.items():
        if len(v) < 3:
            continue
        sl = np.array([a for a, _, _ in v])
        if np.subtract(*np.percentile(sl, [75, 25])) > 0.1:   # tutarsiz: sistematik degil
            continue
        d = np.median([c for _, _, c in v], axis=0); d /= np.linalg.norm(d)
        drift[key] = dict(slope=float(np.median(sl)), pivot=np.median([p for _, p, _ in v], axis=0), d=d, n=len(v))
    if args.no_drift_correction:
        drift = {}
    print(">>> Sistematik yanal kayma duzeltmesi (kol, kavsaga dogru?): " +
          (", ".join(f"{a}{'-giris' if b else '-cikis'}: {v['slope']:+.3f} m/m (n={v['n']})"
                     for (a, b), v in sorted(drift.items())) or "yok"))

    def correct(k):
        Pc = k["P"].copy()
        for ii in segments(k):
            key = (k["arms"][ii[0]], bool(k["inbound"][ii].mean() > 0.5))
            if key not in drift:
                continue
            dr = drift[key]; dd = dr["d"]; nn = np.array([-dd[1], dd[0]])
            rel = Pc[ii] - dr["pivot"]
            u = rel @ dd
            Pc[ii] = Pc[ii] - (dr["slope"] * u)[:, None] * nn[None, :]
        return Pc

    def assign(Pv, k):
        """(a) eski mantik ve (b) histerezisli kilit; dondurur: state, lane_ids, off, lat, lane_dir, old_seq, new_seq"""
        cos_min = 0.5 if k["never_moved"] else 0.7
        off = np.zeros_like(Pv); state, lane_ids = [], []
        lat = np.full(len(Pv), np.nan); lane_dir = np.full_like(Pv, np.nan)
        locked, out_count, old_lane = None, 0, None
        old_seq, new_seq = [], []
        for i in range(len(Pv)):
            if k["injunc"][i]:
                state.append("junction"); lane_ids.append("")
                locked, out_count, old_lane = None, 0, None
                old_seq.append(None); new_seq.append(None)
                continue
            q = lanes.query(Pv[i], k["head"][i], cos_min)
            near = {l: v for l, v in q.items() if v[0] <= args.snap_max}
            if near:   # (a) eski: aninda en yakin (mevcut 1 m'den fazla uzak degilse korunur)
                b_old = min(near, key=lambda l: near[l][0])
                if old_lane in near and near[old_lane][0] <= near[b_old][0] + 1.0:
                    b_old = old_lane
                old_lane = b_old
            else:
                old_lane = None
            old_seq.append(old_lane)
            if locked is None or locked not in q:   # (b) kilit + histerezis
                locked = min(near, key=lambda l: near[l][0]) if near else None
                out_count = 0
            else:
                dist_l, _, _, beyond_end, before_start = q[locked]
                if beyond_end or before_start:
                    nxt = [l for l in near if l != locked and lanes.connected(locked, l)]
                    if nxt:
                        locked = min(nxt, key=lambda l: near[l][0])
                    elif near:
                        locked = min(near, key=lambda l: near[l][0])
                    out_count = 0
                else:
                    out_count = out_count + 1 if dist_l > lanes.width[locked] / 2 + args.lane_margin else 0
                    if out_count >= hold_steps and near:
                        locked = min(near, key=lambda l: near[l][0])
                        out_count = 0
            if locked is None or q.get(locked, (1e9,))[0] > args.snap_max:
                state.append("no_lane"); lane_ids.append(""); new_seq.append(None)
                continue
            dist_l, proj, ldir, _, _ = q[locked]
            off[i] = proj - Pv[i]; lat[i] = dist_l; lane_dir[i] = ldir
            state.append("snapped"); lane_ids.append(locked); new_seq.append(locked)
        return state, lane_ids, off, lat, lane_dir, old_seq, new_seq

    def n_changes(seq):
        c, prev = 0, None
        for l in seq:
            if l is None:
                prev = None; continue
            if prev is not None and l != prev and not lanes.connected(prev, l):
                c += 1
            prev = l
        return c

    def supported(Pv, ii, ref):
        """ham yanal konum (1 s medyan) bir serit genisligi kadar kayip >= 1.5 s kalirsa bir degisim"""
        if ref is None or len(ii) < 20:
            return 0
        offs = pd.Series([lanes.signed_offset(Pv[i], ref) for i in ii]).rolling(11, center=True, min_periods=1).median()
        lidx = np.round(offs.to_numpy() / lanes.width[ref]).astype(int)
        n, cur, run = 0, lidx[0], 0
        for v in lidx[1:]:
            if v != cur:
                run += 1
                if run >= 15:
                    n += 1; cur = v; run = 0
            else:
                run = 0
        return n

    # ---------------- 3. gecis: serit atama + cikti ----------------
    for k in kin:
        tid, P, K, head, injunc, mv = k["tid"], k["P"], k["K"], k["head"], k["injunc"], k["mv"]
        arms, inbound, spd, cls = k["arms"], k["inbound"], k["spd"], k["cls"]
        cand_in = np.flatnonzero(mv & inbound & ~injunc)
        cand_out = np.flatnonzero(mv & ~inbound & ~injunc)
        origin = arms[cand_in[0]] if len(cand_in) else "?"
        dest = arms[cand_out[-1]] if len(cand_out) else "?"
        o_inf = d_inf = False
        if origin == "?":
            if mv.any():
                j = np.flatnonzero(mv)[0]
                origin = arm_of(*(P[j] - 30 * head[j]), cx, cy); o_inf = True
            elif not injunc[0]:
                origin = arms[0]
        if dest == "?" and mv.any():
            j = np.flatnonzero(mv)[-1]
            dest = arm_of(*(P[j] + 30 * head[j]), cx, cy); d_inf = True

        _, _, _, _, _, old_raw, lock_raw = assign(P, k)          # duzeltmesiz (karsilastirma)
        Pc = correct(k)
        state, lane_ids, off, lat, lane_dir, _, new_seq = assign(Pc, k)
        for ii in segments(k):
            ref_raw = next((lock_raw[i] for i in ii if lock_raw[i]), None)
            ref_c = next((new_seq[i] for i in ii if new_seq[i]), None)
            mv_seg = mv[ii].any()
            lc_rows.append(dict(track_id=tid, arm=arms[ii[0]], steps=len(ii),
                                before=n_changes([old_raw[i] for i in ii]),
                                lock_only=n_changes([lock_raw[i] for i in ii]),
                                after=n_changes([new_seq[i] for i in ii]),
                                raw_supported=supported(P, ii, ref_raw) if mv_seg else 0,
                                raw_supported_corr=supported(Pc, ii, ref_c) if mv_seg else 0))

        off_s = pd.DataFrame(off).rolling(10, center=True, min_periods=1).mean().to_numpy()
        X = Pc + off_s
        dvec = np.where(np.isfinite(lane_dir), lane_dir, head)
        ang = np.degrees(np.unwrap(np.arctan2(dvec[:, 0], dvec[:, 1])))
        ang = pd.Series(ang).rolling(5, center=True, min_periods=1).mean().to_numpy() % 360
        for i in range(len(P)):
            rows.append((int(K[i]), round(K[i] * STEP, 1), int(tid), cls, origin, dest, o_inf, d_inf,
                         round(P[i, 0], 2), round(P[i, 1], 2), round(X[i, 0], 2), round(X[i, 1], 2),
                         round(float(ang[i]), 1), round(float(spd[i]), 2), arms[i], bool(inbound[i]),
                         bool(injunc[i]), state[i], lane_ids[i], None if np.isnan(lat[i]) else round(float(lat[i]), 2)))
    cols = ["step", "time_s", "track_id", "cls", "origin", "dest", "origin_inferred", "dest_inferred",
            "raw_x", "raw_y", "x", "y", "angle", "speed", "arm", "inbound", "in_junction", "snap_state", "lane_id",
            "lat_shift"]
    out = pd.DataFrame(rows, columns=cols).sort_values(["step", "track_id"])
    out.to_csv(args.out, index=False)

    # ------------- rapor -------------
    print(f">>> Gecerli bolge (<= {args.max_dist:.0f} m): {n_removed}/{n_total_steps} arac-adimi atildi "
          f"({n_removed / max(n_total_steps, 1):.1%})")
    print(f">>> {out.track_id.nunique()} arac, {len(out)} arac-adimi -> {args.out}")
    tr = out.drop_duplicates("track_id")
    print(">>> Koken (hareketten):", {str(k): int(v) for k, v in tr.origin.value_counts().items()},
          f"(bunlardan {int(tr.origin_inferred.sum())} tanesi ilk hareket yonunden cikarildi)")
    arm_rows = []
    for arm in "NESW":
        a = out[(out.arm == arm) & ~out.in_junction]
        sn = a[a.snap_state == "snapped"]
        arm_rows.append(dict(kol=arm, adim=len(a), oturtulan=f"{len(sn) / max(len(a), 1):.1%}",
                             serit_yok=int((a.snap_state == "no_lane").sum()),
                             kayma_medyan_m=round(sn.lat_shift.median(), 2) if len(sn) else None,
                             kayma_p90_m=round(sn.lat_shift.quantile(0.9), 2) if len(sn) else None))
    print(">>> Serit oturtma (kavsak kutusu disindaki adimlar):")
    print(pd.DataFrame(arm_rows).to_string(index=False))
    lc = pd.DataFrame(lc_rows)
    lct = lc.groupby("arm")[["before", "lock_only", "after", "raw_supported", "raw_supported_corr"]].sum().reindex(list("NESW"), fill_value=0)
    lct["kol_parcasi"] = lc.groupby("arm").size().reindex(list("NESW"), fill_value=0)
    print(f">>> SUMO'daki serit degisimi sayisi (marj {args.lane_margin} m, bekleme {args.lane_hold} s); "
          f"ardil seride gecis sayilmaz:")
    print(lct.rename(columns={"before": "once (eski)", "lock_only": "kilit (duzeltmesiz)",
                              "after": "sonra (kilit+kayma duz.)", "raw_supported": "ham iz destekler",
                              "raw_supported_corr": "ham iz destekler (kayma duz.)"}).to_string())
    id_map = {str(k): {"root": int(v), "status": ("mirror" if v in set(out.track_id) else
                                                   "short" if v in set(short) else "outside")}
              for k, v in root_map.items()}
    id_map.update({str(k): {"root": int(k), "status": "static"} for k in static0})
    with open(os.path.join(os.path.dirname(args.out), "mirror_track_ids.json"), "w") as f:
        json.dump(id_map, f)
    with open(args.stats_out, "w") as f:
        json.dump({"duplicate_boxes_removed": n_dup, "static_tracks_safety_net": [int(x) for x in static0],
                   "stitches": n_st, "stitch_rejected": n_rej, "short_tracks": int(len(short)),
                   "removed_steps": n_removed, "total_steps": n_total_steps, "max_dist_m": args.max_dist,
                   "snap_max_m": args.snap_max, "lane_margin_m": args.lane_margin, "lane_hold_s": args.lane_hold,
                   "drift_correction": {f"{a}_{'in' if b else 'out'}": {"slope": v["slope"], "n": v["n"]}
                                        for (a, b), v in drift.items()},
                   "per_arm": arm_rows, "lane_changes": lct.reset_index().to_dict("records"),
                   "origins": {str(k): int(v) for k, v in tr.origin.value_counts().items()}}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
