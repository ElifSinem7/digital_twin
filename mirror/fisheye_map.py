#!/usr/bin/env python3
"""
fisheye_map.py
================
Kamera goruntusu (piksel) <-> SUMO zemin duzlemi (metre) donusumu.

Model:  balikgozu "division" modeli + homografi
    q   = (p - c) / F                       (normalize piksel, c = bozulma merkezi)
    q_u = q / (1 + k1*|q|^2 + k2*|q|^4)     (bozulmasi giderilmis)
    w   = H * [q_u, 1]                      (SUMO x, y)

Ters donusum (SUMO -> piksel) H^-1 ve yaricap uzerinde ikiye bolme
(bisection) ile yapilir; goruntu disinda kalan noktalar NaN doner.
Sadece numpy gerektirir.
"""
import json

import numpy as np

F = 800.0  # normalizasyon olcegi (piksel)


class FisheyeGroundMap:
    def __init__(self, cx, cy, k1, k2, H, img_size=(1280, 720)):
        self.cx, self.cy, self.k1, self.k2 = float(cx), float(cy), float(k1), float(k2)
        self.H = np.asarray(H, float).reshape(3, 3)
        self.Hinv = np.linalg.inv(self.H)
        self.img_size = img_size
        w, h = img_size
        corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], float)
        self.r_max = float(np.max(np.hypot(corners[:, 0] - cx, corners[:, 1] - cy)) / F)
        # f(r) = r / (1+k1 r^2+k2 r^4) monoton artan kalmali: gecerli yaricap siniri
        rs = np.linspace(0, self.r_max * 1.05, 2000)
        den = 1 + self.k1 * rs ** 2 + self.k2 * rs ** 4
        f = rs / np.maximum(den, 1e-9)
        bad = (den < 0.05) | (np.diff(f, prepend=-1) <= 0)
        self.r_lim = float(rs[np.argmax(bad)] if bad[1:].any() else rs[-1])
        # homojen koordinatin isareti: kesin zeminde olan bir noktadan (goruntunun alt ortasi) belirlenir
        qu = self.undistort_norm([[w / 2, h * 0.9]])
        self.sgn = float(np.sign((np.r_[qu[0], 1.0] @ self.H.T)[2]) or 1.0)

    # ---------- serilestirme ----------
    @classmethod
    def from_params(cls, p, img_size=(1280, 720)):
        cx, cy, k1, k2 = p[:4]
        H = np.append(p[4:12], 1.0)
        return cls(cx, cy, k1, k2, H, img_size)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["cx"], d["cy"], d["k1"], d["k2"], d["H"], tuple(d.get("img_size", (1280, 720))))

    def save(self, path, **extra):
        d = {"model": "division-fisheye + homography", "F": F, "cx": self.cx, "cy": self.cy,
             "k1": self.k1, "k2": self.k2, "H": self.H.tolist(), "img_size": list(self.img_size)}
        d.update(extra)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    # ---------- donusumler ----------
    def undistort_norm(self, pts):
        p = np.atleast_2d(np.asarray(pts, float))
        q = (p - [self.cx, self.cy]) / F
        r2 = (q ** 2).sum(1, keepdims=True)
        return q / (1 + self.k1 * r2 + self.k2 * r2 ** 2)

    def img_to_world(self, pts):
        """(N,2) piksel -> (N,2) SUMO metre. Ufkun otesi / gecersiz yaricap -> NaN."""
        p = np.atleast_2d(np.asarray(pts, float))
        qu = self.undistort_norm(p)
        hw = np.c_[qu, np.ones(len(qu))] @ self.H.T
        w = hw[:, :2] / hw[:, 2:3]
        r = np.hypot(p[:, 0] - self.cx, p[:, 1] - self.cy) / F
        w[(r > self.r_lim) | (hw[:, 2] * self.sgn <= 0)] = np.nan
        return w

    def world_to_img(self, pts):
        """(N,2) SUMO metre -> (N,2) piksel (gorus alani disi -> NaN)."""
        w = np.atleast_2d(np.asarray(pts, float))
        hq = np.c_[w, np.ones(len(w))] @ self.Hinv.T
        ok = hq[:, 2] * self.sgn > 0
        qu = hq[:, :2] / hq[:, 2:3]
        ru = np.hypot(qu[:, 0], qu[:, 1])
        lo, hi = np.zeros_like(ru), np.full_like(ru, self.r_lim)
        f = lambda r: r / (1 + self.k1 * r ** 2 + self.k2 * r ** 4)
        ok &= ru <= f(self.r_lim)
        for _ in range(50):
            mid = (lo + hi) / 2
            m = f(mid) < ru
            lo, hi = np.where(m, mid, lo), np.where(m, hi, mid)
        rd = (lo + hi) / 2
        scale = np.where(ru > 1e-12, rd / np.maximum(ru, 1e-12), 1.0)
        p = qu * scale[:, None] * F + [self.cx, self.cy]
        p[~ok] = np.nan
        return p
