#!/usr/bin/env python3
"""
fit_mapping.py
================
control_points.json'daki nokta + cizgi eslesmelerinden goruntu -> SUMO zemin
donusumunu (fisheye_map.FisheyeGroundMap) kestirir, birakip-birini-dogrula
(leave-one-out) ile metre cinsinden hatayi raporlar, farkli modelleri
karsilastirir ve kontrol gorsellerini cizer.

Cikti:
  mirror/fisheye_mapping.json                 (secilen model)
  demo/mapping_control_points_video.png       (video karesi + noktalar + SUMO seritleri geri izdusumu)
  demo/mapping_control_points_sumo.png        (SUMO haritasi + noktalar + LOO hatalari + kamera gorus alani)

Kullanim (proje venv'i, SUMO_HOME tanimli):
    python fit_mapping.py
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from matplotlib.path import Path

from fisheye_map import F, FisheyeGroundMap

if "SUMO_HOME" not in os.environ:
    sys.exit("HATA: SUMO_HOME tanimli degil.")
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import sumolib  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
JUNCTION = "cluster_53190763_5896114911_7619513267_7619513268_#3more"
W_LINE = 0.7  # cizgi orneklerinin agirligi (nokta artiklarina gore)


# ------------------------------------------------------------------ geometri yardimcilari
def offset_poly(shape, d):
    sh = np.asarray(shape, float)
    out = []
    for i in range(len(sh)):
        a, b = sh[max(i - 1, 0)], sh[min(i + 1, len(sh) - 1)]
        t = (b - a) / (np.linalg.norm(b - a) + 1e-12)
        out.append(sh[i] + np.array([-t[1], t[0]]) * d)  # +d = sol taraf
    return np.array(out)


def is_road_lane(ln):
    return ln.allows("passenger") or (ln.allows("bicycle") and not ln.allows("pedestrian"))


def border_polyline(net, edges, which):
    pts = []
    for eid in edges:
        lanes = [l for l in net.getEdge(eid).getLanes() if is_road_lane(l)]
        if which == "left_of_leftmost":
            ln = max(lanes, key=lambda l: l.getIndex()); d = ln.getWidth() / 2
        else:
            ln = min(lanes, key=lambda l: l.getIndex()); d = -ln.getWidth() / 2
        pts.extend(offset_poly(ln.getShape(), d))
    return np.array(pts)


def dist_to_polyline(P, L):
    P = np.atleast_2d(P)
    A, B = L[:-1], L[1:]
    AB = B - A
    t = ((P[:, None, :] - A[None]) * AB[None]).sum(-1) / np.maximum((AB ** 2).sum(-1), 1e-12)[None]
    t = np.clip(t, 0, 1)
    proj = A[None] + t[..., None] * AB[None]
    return np.sqrt(((P[:, None, :] - proj) ** 2).sum(-1)).min(1)


def densify(px, step=8.0):
    px = np.asarray(px, float); out = [px[0]]
    for a, b in zip(px[:-1], px[1:]):
        n = max(1, int(np.linalg.norm(b - a) / step))
        out += [a + (b - a) * k / n for k in range(1, n + 1)]
    return np.array(out)


# ------------------------------------------------------------------ Levenberg-Marquardt (numpy)
def lm(fun, p0, free, iters=400):
    p = np.asarray(p0, float).copy()
    idx = np.flatnonzero(free)
    r = fun(p); cost = r @ r; lam = 1e-3
    for _ in range(iters):
        J = np.empty((len(r), len(idx)))
        for j, k in enumerate(idx):
            h = 1e-6 * max(1.0, abs(p[k]))
            pp, pm = p.copy(), p.copy(); pp[k] += h; pm[k] -= h
            J[:, j] = (fun(pp) - fun(pm)) / (2 * h)
        A = J.T @ J; g = J.T @ r
        improved = False
        for _ in range(12):
            try:
                step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            except np.linalg.LinAlgError:
                lam *= 10; continue
            pn = p.copy(); pn[idx] += step
            rn = fun(pn); cn = rn @ rn
            if np.isfinite(cn) and cn < cost:
                p, r, lam, improved = pn, rn, lam * 0.3, True
                if cost - cn < 1e-12 * max(cost, 1):
                    return p
                cost = cn
                break
            lam *= 10
        if not improved:
            break
    return p


def init_params(px, xy, cx=640.0, cy=360.0):
    q = (np.asarray(px, float) - [cx, cy]) / F
    H, _ = cv2.findHomography(q, np.asarray(xy, float), 0)
    H = H / H[2, 2]
    return np.r_[cx, cy, 0.0, 0.0, H.ravel()[:8]]


MODELS = {
    # ad: (serbest parametre maskesi [cx,cy,k1,k2,H*8], cizgiler kullanilsin mi)
    "H (sadece nokta)":            ([0, 0, 0, 0] + [1] * 8, False),
    "H + cizgiler":                ([0, 0, 0, 0] + [1] * 8, True),
    "balikgozu k1 + H + cizgiler": ([0, 0, 1, 0] + [1] * 8, True),
    "balikgozu k1,k2,merkez + H + cizgiler": ([1, 1, 1, 1] + [1] * 8, True),
}


def fit(px, xy, lines, free, use_lines):
    px, xy = np.asarray(px, float), np.asarray(xy, float)

    def res(p):
        m = FisheyeGroundMap.from_params(p)
        w = m.img_to_world(px)
        r = [(w - xy).ravel()]
        if use_lines:
            for L in lines:
                d = dist_to_polyline(m.img_to_world(L["samples"]), L["poly"])
                r.append(W_LINE * d / np.sqrt(len(d) / 3))
        if free[0]:
            r.append(np.array([(p[0] - 640) / 150, (p[1] - 360) / 150]))
        if free[3]:
            r.append(np.array([p[3] / 2.0]))
        out = np.concatenate(r)
        return np.where(np.isfinite(out), out, 1e3)

    p = init_params(px, xy)
    if free[2]:
        # k1 icin birkac baslangic dene (yerel minimuma karsi)
        best = None
        for k1 in (0.0, -0.1, -0.2, -0.35):
            p0 = p.copy(); p0[2] = k1
            pf = lm(res, p0, np.array(free, bool))
            c = (res(pf) ** 2).sum()
            if best is None or c < best[0]:
                best = (c, pf)
        return best[1]
    return lm(res, p, np.array(free, bool))


# ------------------------------------------------------------------ dogru-cizgi (plumb-line) kalibrasyonu
# Gercekte DUZ olan yaya gecidi cizgileri (uc noktalari videodan); goruntudeki egrilikleri
# yalnizca mercek bozulmasindan gelir -> k1,k2 SUMO geometrisinden BAGIMSIZ kestirilir.
PLUMB_LINES = {
    "W dis":  [(403, 253), (507, 673)], "W ic": [(452, 228), (641, 668)],
    "N ic":   [(451, 226), (757, 159)], "N dis": [(452, 204), (717, 149)],
    "E ic":   [(757, 159), (1046, 352)], "E dis": [(787, 157), (1035, 312)],
    "S ic":   [(641, 668), (1046, 352)],
}


def plumb_samples(frame, n=30, half=14):
    """Her cizgi boyunca, dik dogrultuda parlaklik profilinden beyaz cizginin merkezini bulur."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
    out = {}
    for name, (a, b) in PLUMB_LINES.items():
        a, b = np.array(a, float), np.array(b, float)
        t = (b - a) / np.linalg.norm(b - a); nrm = np.array([-t[1], t[0]])
        pts = []
        for s in np.linspace(0.06, 0.94, n):
            c = a + (b - a) * s
            offs = np.arange(-half, half + 1)
            prof = np.array([cv2.getRectSubPix(gray.astype(np.float32), (1, 1), tuple(c + o * nrm))[0, 0] for o in offs])
            thr = prof.max() - 0.35 * (prof.max() - np.median(prof))
            m = prof >= thr
            if prof.max() - np.median(prof) < 40 or m.sum() == 0:
                continue  # cizgi gorunmuyor (arac/golge)
            o = (offs[m] * (prof[m] - thr)).sum() / (prof[m] - thr).sum()
            pts.append(c + o * nrm)
        out[name] = np.array(pts)
    return out


def straightness_residuals(k, samples, cx=640.0, cy=360.0):
    m = FisheyeGroundMap(cx, cy, k[0], k[1], np.eye(3))
    r = []
    for pts in samples.values():
        q = m.undistort_norm(pts) * F
        mu = q.mean(0); _, _, vt = np.linalg.svd(q - mu)
        r.append((q - mu) @ vt[1])  # dogruya dik uzaklik (piksel olcegi)
    return np.concatenate(r)


def fit_plumb(samples):
    best = None
    for k1 in (0.0, -0.1, -0.25):
        k = lm(lambda p: straightness_residuals(p, samples), np.array([k1, 0.0]), np.array([True, True]))
        c = (straightness_residuals(k, samples) ** 2).sum()
        if best is None or c < best[0]:
            best = (c, k)
    return best[1]


def fit_h_given_k(px, xy, k, lines=None):
    """k1,k2 sabit; sadece homografi (istege bagli cizgilerle)."""
    p0 = init_params(px, xy); p0[2:4] = k
    q = FisheyeGroundMap(640, 360, k[0], k[1], np.eye(3)).undistort_norm(px)
    H, _ = cv2.findHomography(q, np.asarray(xy, float), 0)
    p0[4:] = (H / H[2, 2]).ravel()[:8]
    free = np.array([0, 0, 0, 0] + [1] * 8, bool)
    px_, xy_ = np.asarray(px, float), np.asarray(xy, float)

    def res(p):
        m = FisheyeGroundMap.from_params(p)
        r = [(m.img_to_world(px_) - xy_).ravel()]
        for L in (lines or []):
            d = dist_to_polyline(m.img_to_world(L["samples"]), L["poly"])
            r.append(W_LINE * d / np.sqrt(len(d) / 3))
        out = np.concatenate(r)
        return np.where(np.isfinite(out), out, 1e3)
    return lm(res, p0, free)


def tps_fit_predict(src, dst, query, lam=1e-3):
    src, dst, query = map(lambda a: np.asarray(a, float), (src, dst, query))
    s = 1000.0
    S, Q = src / s, query / s
    U = lambda r: np.where(r > 0, r ** 2 * np.log(r + 1e-300), 0)
    K = U(np.linalg.norm(S[:, None] - S[None], axis=-1))
    P = np.c_[np.ones(len(S)), S]
    n = len(S)
    A = np.zeros((n + 3, n + 3)); A[:n, :n] = K + lam * np.eye(n); A[:n, n:] = P; A[n:, :n] = P.T
    b = np.zeros((n + 3, 2)); b[:n] = dst
    sol = np.linalg.solve(A, b)
    Kq = U(np.linalg.norm(Q[:, None] - S[None], axis=-1))
    return Kq @ sol[:n] + np.c_[np.ones(len(Q)), Q] @ sol[n:]


# ------------------------------------------------------------------ cizim
def grab_frame(video, t):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    for _ in range(int(round(t * fps))):
        cap.grab()
    ok, fr = cap.read()
    cap.release()
    if not ok:
        sys.exit("HATA: kare okunamadi")
    return fr


def lane_polys(net, X0, X1, Y0, Y1, internal=True):
    out = []
    for e in net.getEdges(withInternal=internal):
        if e.getFunction() in ("crossing", "walkingarea"):
            continue
        for ln in e.getLanes():
            if not is_road_lane(ln):
                continue
            sh = ln.getShape()
            if not any(X0 < x < X1 and Y0 < y < Y1 for x, y in sh):
                continue
            w = ln.getWidth()
            out.append((e.getFunction() == "internal",
                        np.vstack([offset_poly(sh, -w / 2), offset_poly(sh, w / 2)[::-1]])))
    return out


def resample(poly, step=0.5):
    return densify(poly, step)


class LaneDirectionCheck:
    """Hareketli araclarin izdusumu, ayni yone akan bir SUMO seridine mi dusuyor?
    (kavsak kutusu disindaki noktalar; serit yonu = serit seklinin en yakin parcasinin yonu)"""

    def __init__(self, net, boxes_csv, X0=500, X1=720, Y0=720, Y1=900, frame_step=3):
        self.lanes = []
        for e in net.getEdges(withInternal=False):
            for ln in e.getLanes():
                if not ln.allows("passenger"):
                    continue
                sh = np.array(ln.getShape())
                if not any(X0 < x < X1 and Y0 < y < Y1 for x, y in sh):
                    continue
                w = ln.getWidth()
                poly = np.vstack([offset_poly(sh, -w / 2), offset_poly(sh, w / 2)[::-1]])
                self.lanes.append((Path(poly), sh))
        self.junction = Path(np.array(net.getNode(JUNCTION).getShape()))
        bx = pd.read_csv(boxes_csv)
        g = bx.groupby("track_id")
        span = np.hypot(g.bx.max() - g.bx.min(), g.by.max() - g.by.min())
        self.b = bx[bx.track_id.isin(span[span > 40].index)].sort_values(["track_id", "frame"])
        self.step = frame_step

    def evaluate(self, m):
        hits = tot = 0
        for _, t in self.b.groupby("track_id"):
            w = m.img_to_world(t[["bx", "by"]].to_numpy())
            w = pd.DataFrame(w).rolling(9, center=True, min_periods=3).mean().to_numpy()
            d = np.full_like(w, np.nan)
            d[7:-7] = w[14:] - w[:-14]  # ~0.5 s ileri-geri fark
            for i in range(0, len(w), self.step):
                p, v = w[i], d[i]
                if not (np.isfinite(p).all() and np.isfinite(v).all()) or np.hypot(*v) < 1.0:
                    continue  # duruyor ya da gecersiz
                if self.junction.contains_point(p):
                    continue
                tot += 1
                for path, sh in self.lanes:
                    if path.contains_point(p):
                        seg = np.argmin(np.linalg.norm(sh[:-1] + np.diff(sh, axis=0) / 2 - p, axis=1))
                        sd = sh[seg + 1] - sh[seg]
                        if (sd @ v) / (np.linalg.norm(sd) * np.linalg.norm(v)) > 0.7:
                            hits += 1
                            break
        return hits / max(tot, 1), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(ROOT, "videos", "Bellevue_116th_NE12th__2017-09-10_19-08-25.mp4"))
    ap.add_argument("--net", default=os.path.join(ROOT, "network", "bellevue_116th_ne12th.net.xml"))
    ap.add_argument("--control", default=os.path.join(HERE, "control_points.json"))
    ap.add_argument("--boxes", default=os.path.join(ROOT, "tracks", "window_384_504_boxes.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "fisheye_mapping.json"))
    ap.add_argument("--demo-dir", default=os.path.join(ROOT, "demo"))
    ap.add_argument("--line-weight", type=float, default=0.7,
                    help="Cizgi kisitlarinin nokta artiklarina gore agirligi")
    args = ap.parse_args()
    globals()["W_LINE"] = args.line_weight

    net = sumolib.net.readNet(args.net, withInternal=True, withPedestrianConnections=True)
    ctl = json.load(open(args.control))
    pts = ctl["points"]
    px = np.array([p["px"] for p in pts], float)
    xy = np.array([p["xy"] for p in pts], float)
    lines = []
    for L in ctl["lines"]:
        lines.append({"name": L["name"], "px": np.array(L["px"], float), "samples": densify(L["px"]),
                      "poly": border_polyline(net, L["edges"], L["border"])})

    frame = grab_frame(args.video, ctl["frame_time_s"])

    # ---------------- 1) mercek bozulmasi: dogru-cizgi kalibrasyonu (SUMO'dan bagimsiz) ----------------
    samples = plumb_samples(frame)
    k_pl = fit_plumb(samples)
    r0 = straightness_residuals([0, 0], samples); r1 = straightness_residuals(k_pl, samples)
    print(f">>> Dogru-cizgi kalibrasyonu: {len(samples)} gecit cizgisi, {sum(len(v) for v in samples.values())} ornek")
    print(f"    k1={k_pl[0]:.4f} k2={k_pl[1]:.4f}  cizgilerden sapma RMS: "
          f"duzeltmesiz {np.sqrt((r0 ** 2).mean()):.2f} px -> duzeltilmis {np.sqrt((r1 ** 2).mean()):.2f} px")
    for nm, v in samples.items():
        a0 = straightness_residuals([0, 0], {nm: v}); a1 = straightness_residuals(k_pl, {nm: v})
        print(f"      {nm:6s} n={len(v):2d}  maks sapma {np.abs(a0).max():5.1f} px -> {np.abs(a1).max():4.1f} px")

    # ---------------- 2) model karsilastirmasi (LOO) ----------------
    print(f"\n>>> {len(pts)} kontrol noktasi, {len(lines)} kontrol cizgisi "
          f"({sum(len(L['samples']) for L in lines)} piksel ornegi)\n")
    ext = pd.read_csv(os.path.join(ROOT, "points.csv"), header=None).drop_duplicates().to_numpy(float)
    fitters = {name: (lambda P_, X_, f=free, u=use_lines: fit(P_, X_, lines, f, u))
               for name, (free, use_lines) in MODELS.items()}
    fitters["balikgozu (dogru-cizgi k1,k2) + H"] = lambda P_, X_: fit_h_given_k(P_, X_, k_pl)
    fitters["balikgozu (dogru-cizgi k1,k2) + H + cizgiler"] = lambda P_, X_: fit_h_given_k(P_, X_, k_pl, lines)
    results = {}
    ldc = LaneDirectionCheck(net, args.boxes)
    print(f"{'model':46s} {'fit RMS':>8s} {'LOO ort':>8s} {'LOO medyan':>10s} {'LOO maks':>9s} "
          f"{'cizgi ort':>9s} {'points.csv ort':>14s} {'dogru-yon serit':>15s}")
    for name, fitter in fitters.items():
        p_all = fitter(px, xy)
        m = FisheyeGroundMap.from_params(p_all)
        fit_err = np.linalg.norm(m.img_to_world(px) - xy, axis=1)
        loo = []
        for i in range(len(pts)):
            keep = np.arange(len(pts)) != i
            mi = FisheyeGroundMap.from_params(fitter(px[keep], xy[keep]))
            loo.append(np.linalg.norm(mi.img_to_world(px[i:i + 1])[0] - xy[i]))
        loo = np.nan_to_num(np.array(loo), nan=99)
        line_err = np.concatenate([dist_to_polyline(m.img_to_world(L["samples"]), L["poly"]) for L in lines])
        ext_err = np.linalg.norm(m.img_to_world(ext[:, :2]) - ext[:, 2:], axis=1)
        lane_ok, n_lane = ldc.evaluate(m)
        results[name] = dict(p=p_all, loo=loo, fit=fit_err, line=line_err, ext=ext_err, fitter=fitter, lane=lane_ok)
        print(f"{name:46s} {np.sqrt((fit_err ** 2).mean()):8.2f} {loo.mean():8.2f} {np.median(loo):10.2f} "
              f"{loo.max():9.2f} {np.nanmean(line_err):9.2f} {np.nanmean(ext_err):14.2f} {lane_ok:14.1%}")
    print(f"    (dogru-yon serit: kavsak kutusu disindaki {n_lane} hareketli arac noktasindan, ayni yone akan "
          f"SUMO seridine dusenlerin orani)")
    # TPS (sadece noktalar; cizgi kisiti eklenemez)
    loo = np.array([np.linalg.norm(tps_fit_predict(np.delete(px, i, 0), np.delete(xy, i, 0), px[i:i + 1])[0] - xy[i])
                    for i in range(len(pts))])
    ext_err = np.linalg.norm(tps_fit_predict(px, xy, ext[:, :2]) - ext[:, 2:], axis=1)
    print(f"{'ince-plaka spline (TPS, sadece nokta)':42s} {0:8.2f} {loo.mean():8.2f} {np.median(loo):10.2f} "
          f"{loo.max():9.2f} {'-':>9s} {ext_err.mean():14.2f}")
    old = json.load(open(os.path.join(ROOT, "homography.json")))["homography_matrix"]
    old_w = cv2.perspectiveTransform(px.reshape(-1, 1, 2).astype(np.float32), np.array(old)).reshape(-1, 2)
    print(f"{'(eski homography.json, bu noktalarda)':42s} {np.sqrt((np.linalg.norm(old_w - xy, axis=1) ** 2).mean()):8.2f}")

    # Secim: LOO ortalamasi en iyiye 0.3 m'den yakin olan modeller arasindan, araclari en cok
    # dogru yondeki SUMO seridine oturtan model (ayna modu icin asil onemli olan bu).
    best_loo = min(r["loo"].mean() for r in results.values())
    cands = [n for n, r in results.items() if r["loo"].mean() <= best_loo + 0.3]
    best_name = max(cands, key=lambda n: results[n]["lane"])
    B = results[best_name]
    m = FisheyeGroundMap.from_params(B["p"])
    print(f"\n>>> Secilen model: {best_name}")
    print(f"    cx={m.cx:.1f} cy={m.cy:.1f} k1={m.k1:.4f} k2={m.k2:.4f}  (gecerli yaricap siniri {m.r_lim * F:.0f} px)")
    print(f"\n{'#':>3s} {'nokta':34s} {'fit hata (m)':>12s} {'LOO hata (m)':>12s}")
    for p, fe, le in zip(pts, B["fit"], B["loo"]):
        print(f"{p['id']:3d} {p['name']:34s} {fe:12.2f} {le:12.2f}")
    print(f"\n    Cizgi ornekleri: ort {np.nanmean(B['line']):.2f} m, maks {np.nanmax(B['line']):.2f} m")
    for L in lines:
        d = dist_to_polyline(m.img_to_world(L["samples"]), L["poly"])
        print(f"      {L['name']:22s} ort {np.nanmean(d):.2f} m  maks {np.nanmax(d):.2f} m")
    m.save(args.out, selected=best_name, loo_mean_m=float(B["loo"].mean()), loo_max_m=float(B["loo"].max()),
           fit_rms_m=float(np.sqrt((B["fit"] ** 2).mean())), n_points=len(pts), n_lines=len(lines))
    print(f">>> Kaydedildi: {args.out}")

    # ---------------- serit isabeti: tespit edilen araclar SUMO seridine dusuyor mu? ----------------
    X0, X1, Y0, Y1 = 500, 720, 720, 900
    road = [Path(pl) for _, pl in lane_polys(net, X0, X1, Y0, Y1)]
    jshape = net.getNode(JUNCTION).getShape()
    road.append(Path(np.array(jshape)))

    def on_road(w):
        w = w[np.isfinite(w).all(1)]
        if len(w) == 0:
            return np.nan, 0
        inside = np.zeros(len(w), bool)
        for pth in road:
            inside |= pth.contains_points(w, radius=0.0)
        return inside.mean(), len(w)

    if os.path.exists(args.boxes):
        bx = pd.read_csv(args.boxes)
        g = bx.groupby("track_id")
        span = np.hypot(g.bx.max() - g.bx.min(), g.by.max() - g.by.min())
        moving = span[span > 40].index
        b = bx[bx.track_id.isin(moving)]
        b = b[b.frame % 3 == 0]
        r_new, n_new = on_road(m.img_to_world(b[["bx", "by"]].to_numpy()))
        r_ctr, _ = on_road(m.img_to_world(np.c_[(b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2]))
        oldt = pd.read_csv(os.path.join(ROOT, "tracks", "2017-09-10_19_stitched_real.csv"))
        oldt = oldt[(oldt.time_s >= 384) & (oldt.time_s < 504)]
        r_old, n_old = on_road(oldt[["real_x", "real_y"]].to_numpy())
        print(f"\n>>> Serit isabeti (384-504 s, hareketli araclar, SUMO arac seridi ya da kavsak alani icinde):")
        print(f"    YENI: kutu alt-ortasi + balikgozu modeli : {r_new:.1%}  ({n_new} nokta)")
        print(f"    kutu merkezi + balikgozu modeli          : {r_ctr:.1%}")
        print(f"    ESKI: merkez + homography.json           : {r_old:.1%}  ({n_old} nokta)")

    # ---------------- gorsel 1: video karesi ----------------
    img = frame.copy()
    for v in samples.values():
        for s in v:
            cv2.circle(img, tuple(np.int32(s)), 2, (0, 255, 0), -1)
    for internal, pl in lane_polys(net, X0, X1, Y0, Y1):
        q = m.world_to_img(resample(np.vstack([pl, pl[:1]])))
        good = np.isfinite(q).all(1)
        for a, b2, ok1, ok2 in zip(q[:-1], q[1:], good[:-1], good[1:]):
            if ok1 and ok2 and np.hypot(*(a - b2)) < 60:
                cv2.line(img, tuple(np.int32(a)), tuple(np.int32(b2)),
                         (0, 200, 255) if not internal else (0, 120, 160), 1, cv2.LINE_AA)
    for e in net.getEdges(withInternal=True):
        if e.getFunction() == "crossing" and JUNCTION in e.getID():
            ln = e.getLanes()[0]; w = ln.getWidth(); sh = ln.getShape()
            poly = np.vstack([offset_poly(sh, -w / 2), offset_poly(sh, w / 2)[::-1]])
            q = m.world_to_img(resample(np.vstack([poly, poly[:1]]), 0.3))
            q = q[np.isfinite(q).all(1)]
            cv2.polylines(img, [np.int32(q)], False, (255, 0, 255), 2, cv2.LINE_AA)
    for L in lines:
        for s in L["samples"]:
            cv2.circle(img, tuple(np.int32(s)), 2, (255, 255, 0), -1)
    back = m.world_to_img(xy)
    for p, pb, le in zip(pts, back, B["loo"]):
        c = tuple(int(v) for v in p["px"])
        cv2.circle(img, c, 7, (0, 0, 255), 2)
        if np.isfinite(pb).all():
            cv2.drawMarker(img, tuple(np.int32(pb)), (0, 255, 0), cv2.MARKER_CROSS, 10, 2)
        cv2.putText(img, f"{p['id']}", (c[0] + 8, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(img, f"{p['id']}", (c[0] + 8, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    leg = [f"Kirmizi daire: kontrol noktasi (videoda secilen)   Yesil +: SUMO koordinatinin modelle geri izdusumu",
           f"Turuncu: SUMO arac seritleri, mor: SUMO yaya gecitleri, camgobegi: kaldirim/orta cizgi ornekleri",
           f"Model: {best_name}   LOO hata ort {B['loo'].mean():.2f} m, maks {B['loo'].max():.2f} m"]
    bar = np.full((70, img.shape[1], 3), 25, np.uint8)
    for i, t in enumerate(leg):
        cv2.putText(bar, t, (10, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    out1 = os.path.join(args.demo_dir, "mapping_control_points_video.png")
    cv2.imwrite(out1, np.vstack([img, bar]))

    # ---------------- gorsel 2: SUMO haritasi ----------------
    sat = cv2.imread(os.path.join(ROOT, "sumo", "satellite.png"))
    SH, SW_ = sat.shape[:2]
    MX0, MX1, MY0, MY1, PX = 545, 665, 770, 870, 9
    xs = np.arange(MX0, MX1, 1 / PX); ys = np.arange(MY1, MY0, -1 / PX)
    gx, gy = np.meshgrid(xs / (1053.36 / SW_), (1778.70 - ys) / (1778.70 / SH))
    mp = cv2.remap(sat, gx.astype(np.float32), gy.astype(np.float32), cv2.INTER_CUBIC)
    mp = cv2.addWeighted(mp, 0.6, np.zeros_like(mp), 0.4, 0)
    P = lambda x, y: (int(round((x - MX0) * PX)), int(round((MY1 - y) * PX)))
    for internal, pl in lane_polys(net, MX0, MX1, MY0, MY1):
        cv2.polylines(mp, [np.array([P(*v) for v in pl], np.int32)], True,
                      (0, 200, 255) if not internal else (0, 110, 150), 1, cv2.LINE_AA)
    for e in net.getEdges(withInternal=True):
        if e.getFunction() == "crossing" and JUNCTION in e.getID():
            ln = e.getLanes()[0]; w = ln.getWidth(); sh = ln.getShape()
            poly = np.vstack([offset_poly(sh, -w / 2), offset_poly(sh, w / 2)[::-1]])
            cv2.polylines(mp, [np.array([P(*v) for v in poly], np.int32)], True, (255, 0, 255), 2, cv2.LINE_AA)
    # kamera gorus alani: goruntu kenarlari + ic izgara
    W_, H_ = 1280, 720
    border = np.vstack([np.c_[np.linspace(0, W_, 400), np.zeros(400)], np.c_[np.full(300, W_), np.linspace(0, H_, 300)],
                        np.c_[np.linspace(W_, 0, 400), np.full(400, H_)], np.c_[np.zeros(300), np.linspace(H_, 0, 300)]])
    wb = m.img_to_world(border)
    far = np.hypot(wb[:, 0] - 595, wb[:, 1] - 811) > 90
    wb[far] = np.nan
    for a, b2 in zip(wb[:-1], wb[1:]):
        if np.isfinite(a).all() and np.isfinite(b2).all():
            cv2.line(mp, P(*a), P(*b2), (0, 140, 255), 2, cv2.LINE_AA)
    for L in lines:
        w = m.img_to_world(L["samples"])
        for s in w[np.isfinite(w).all(1)]:
            cv2.circle(mp, P(*s), 2, (255, 255, 0), -1)
    for i, p in enumerate(pts):
        keep = np.arange(len(pts)) != i
        mi = FisheyeGroundMap.from_params(B["fitter"](px[keep], xy[keep]))
        pl = mi.img_to_world(px[i:i + 1])[0]
        t = P(*xy[i])
        cv2.circle(mp, t, 6, (0, 0, 255), 2)
        if np.isfinite(pl).all():
            cv2.line(mp, t, P(*pl), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.drawMarker(mp, P(*pl), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 9, 2)
        cv2.putText(mp, str(p["id"]), (t[0] + 7, t[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(mp, str(p["id"]), (t[0] + 7, t[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    for x in range(MX0, MX1 + 1, 10):
        cv2.putText(mp, str(x), (P(x, MY1)[0] + 2, 12), 0, 0.35, (220, 220, 220), 1)
    for y in range(MY0, MY1 + 1, 10):
        cv2.putText(mp, str(y), (2, P(MX0, y)[1] - 2), 0, 0.35, (220, 220, 220), 1)
    cv2.arrowedLine(mp, (mp.shape[1] - 30, 70), (mp.shape[1] - 30, 30), (255, 255, 255), 2, tipLength=0.3)
    cv2.putText(mp, "N", (mp.shape[1] - 36, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    leg = ["Kirmizi daire: SUMO'daki kontrol noktasi   Yesil x: nokta disarida birakilarak (LOO) videodan tahmin edilen yer",
           "Turuncu kalin: kamera gorus alani siniri (goruntu kenarlari, <=90 m)   Camgobegi: cizgi orneklerinin izdusumu",
           f"LOO hata ort {B['loo'].mean():.2f} m, medyan {np.median(B['loo']):.2f} m, maks {B['loo'].max():.2f} m"]
    bar = np.full((70, mp.shape[1], 3), 25, np.uint8)
    for i, t in enumerate(leg):
        cv2.putText(bar, t, (10, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    out2 = os.path.join(args.demo_dir, "mapping_control_points_sumo.png")
    cv2.imwrite(out2, np.vstack([mp, bar]))
    print(f">>> Gorseller: {out1}\n               {out2}")


if __name__ == "__main__":
    main()
