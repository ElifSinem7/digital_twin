#!/usr/bin/env python3
"""
signal_reader.py
==================
Videodaki okunabilir sinyal kafalarindan LAMBA LAMBA durum okur (kutu rengi degil).

Her kafa icin lambalarin goruntudeki konumu elle, buyutulmus kirpintilardan olculdu
(HEADS). Her karede her lamba icin r=2 px'lik diskteki en parlak 5 pikselin V ortalamasi
alinir; lambanin kendi "sonuk" seviyesi (pencere boyunca 10. yuzdelik) cikarilir.
Esigi (+40) gecen en guclu lamba kafanin durumudur; hicbiri gecmiyorsa '-'.
Yanip sonen sari ok (FYA, izinli sola donus) 1.5 s'lik pencerede Y/'-' karisimi olarak
gorulur ve 'F' olarak isaretlenir.

Not: bu kamerada yesil lambalar doygunlugunu kaybedip beyaz-camgobegi gorunur; bu yuzden
renk tonu degil, lambanin KONUMU (ust=kirmizi, orta=sari, alt=yesil) belirleyicidir.
Sokak tabelasi gibi sabit yesil nesneler kafa disinda kaldigi icin etkilemez.

Cikti: --out-timeline (time_s, kafa sutunlari, W_thru, W_left, S_thru, S_left, NW)
       --sheet        5 s arayla kirpintilar + otomatik etiketler (gorsel dogrulama icin)
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# kafa -> [(renk, x, y)] ; olcumler 395/410/430/450/462/480/495 s karelerinden
HEADS = {
    # dogu kolu diregi: batidan gelen (dogu yonlu) trafik
    "Wa": [("R", 978.7, 113.2), ("Y", 977.3, 118.6), ("F", 976.0, 124.0)],  # sola donus kafasi (FYA)
    "Wb": [("R", 1017.3, 136.2), ("Y", 1015.7, 141.6), ("G", 1014.1, 147.0)],
    "Wc": [("R", 1066.7, 168.6), ("Y", 1064.4, 174.0), ("G", 1062.1, 179.5)],
    # kuzey kolu diregi: guneyden gelen (kuzey yonlu) trafik
    "S1": [("R", 553.7, 51.9), ("Y", 553.3, 57.0), ("F", 553.0, 62.0), ("G", 552.0, 68.8)],  # sola donus
    "S2": [("R", 607.0, 53.0), ("Y", 606.6, 58.2), ("G", 606.3, 63.3)],
    "S3": [("R", 650.2, 53.7), ("Y", 650.2, 58.0), ("G", 650.2, 62.0)],
    "S4": [("R", 698.8, 53.0), ("G", 698.8, 61.0)],
    # KB kosedeki direk: hangi yaklasimi kontrol ettigi hareketten dogrulanir
    "NW": [("R", 360.0, 150.0), ("Y", 363.4, 161.7), ("G", 363.5, 167.4)],
}
CROPS = {"W": (955, 100, 1090, 200), "S": (535, 35, 715, 75), "NW": (330, 135, 390, 200)}


def read_frames(video, t0, t1, every=1):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    for _ in range(int(round(t0 * fps))):
        cap.grab()
    i0 = int(round(t0 * fps)); i1 = int(round(t1 * fps))
    for i in range(i0, i1):
        ok, fr = cap.read()
        if not ok:
            break
        if (i - i0) % every == 0:
            yield i / fps, fr
    cap.release()


def lamp_levels(v_channel):
    out = {}
    for h, lamps in HEADS.items():
        for c, x, y in lamps:
            x0, y0 = int(round(x)), int(round(y))
            patch = v_channel[y0 - 2:y0 + 3, x0 - 2:x0 + 3].astype(float).ravel()
            out[(h, c)] = np.sort(patch)[-5:].mean()
    return out


def classify(levels, base, thr=40.0):
    st = {}
    for h, lamps in HEADS.items():
        best, bc = thr, "-"
        for c, _, _ in lamps:
            s = levels[(h, c)] - base[(h, c)]
            if s > best:
                best, bc = s, c
        st[h] = bc
    return st


def majority(vals):
    vals = [v for v in vals if v != "-"]
    if not vals:
        return "-"
    return max(set(vals), key=vals.count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(ROOT, "videos", "Bellevue_116th_NE12th__2017-09-10_19-08-25.mp4"))
    ap.add_argument("--start", type=float, default=384)
    ap.add_argument("--end", type=float, default=504)
    ap.add_argument("--out-timeline", default=os.path.join(HERE, "signal_timeline_384_504.csv"))
    ap.add_argument("--sheet", default=os.path.join(ROOT, "demo", "signal_lamps.png"))
    args = ap.parse_args()

    recs, sheet_frames = [], {}
    for t, fr in read_frames(args.video, args.start, args.end):
        v = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)[..., 2]
        lv = lamp_levels(v); lv["time_s"] = t
        recs.append(lv)
        k = int(round(t))
        if k % 5 == 0 and k not in sheet_frames and abs(t - k) < 0.04:
            sheet_frames[k] = fr
    raw = pd.DataFrame(recs).set_index("time_s")
    base = {c: np.percentile(raw[c], 10) for c in raw.columns}
    heads = pd.DataFrame([classify(r, base) for r in raw.to_dict("records")], index=raw.index)

    # yanip sonen sari ok: 1.5 s'lik pencerede Y/F yanik orani >= %20 ve kirmizi yok -> 'F'
    for h in ("Wa", "S1"):
        roll = lambda s: s.astype(float).rolling(45, center=True, min_periods=1).mean()
        isF = roll(heads[h] == "F")
        isY = roll(heads[h] == "Y")      # sabit sari ok (sola donus fazinin sonu)
        isR = roll(heads[h] == "R")
        isG = roll(heads[h] == "G")
        heads[h] = np.where(isG > 0.5, "G", np.where(isR > 0.5, "R", np.where(isY > 0.5, "Y",
                            np.where(isF >= 0.2, "F", heads[h]))))
    # 0.5 s'lik cogunluk ile kisa okuma hatalarini temizle
    sm = heads.copy()
    for h in heads.columns:
        codes = heads[h].astype("category")
        cats = list(codes.cat.categories)
        oh = pd.get_dummies(codes).astype(float).rolling(15, center=True, min_periods=1).mean()
        sm[h] = [cats[i] for i in oh.to_numpy().argmax(1)] if len(cats) > 1 else heads[h]
    sm["W_thru"] = [majority([a, b]) for a, b in zip(sm.Wb, sm.Wc)]
    sm["W_left"] = sm["Wa"]
    sm["S_thru"] = [majority([a, b]) for a, b in zip(sm.S2, sm.S3)]
    sm["S_left"] = sm["S1"]
    sm.index = sm.index.round(3)
    sm.to_csv(args.out_timeline)
    print(f">>> {len(sm)} kare okundu -> {args.out_timeline}")
    for c in ["W_thru", "W_left", "S_thru", "S_left"]:
        seg = (sm[c] != sm[c].shift()).cumsum()
        runs = sm.groupby(seg).agg(first=(c, "first"), t0=(c, lambda s: s.index[0]))
        txt = " ".join(f"{r['first']}@{r['t0']:.1f}" for _, r in runs.iterrows())
        print(f"    {c:7s}: {txt}")

    # Dogruluk: 5 s arayla kontrol gorselinden GOZLE okunan etiketler (385..500 s)
    gt = {
        "W_thru": "RRGGYRRRGGYRRRRGGGYRRRRR",
        "W_left": "RRFFFFFFFFYRRRRFFFYRRRRR",
        "S_thru": "GYRRRRRRRRRGGGYRRRRGGGGG",
        "S_left": "FYRRRRRRRRRFFFYRRRRGGYFF",
    }
    times = list(range(385, 501, 5))
    print(">>> Gozle etiketlenmis 24 ana gore dogruluk:")
    for c, lab in gt.items():
        auto = [sm[c].iloc[np.argmin(np.abs(sm.index - t))] for t in times]
        ok = sum(a == b for a, b in zip(auto, lab))
        bad = [f"{t}s:{a}!={b}" for t, a, b in zip(times, auto, lab) if a != b]
        print(f"    {c:7s}: {ok}/24  {' '.join(bad)}")
    nw = sm.NW.replace({"Y": "F"}).where(sm.NW != "-", "F")
    sl = sm.S_left.replace({"Y": "F"})
    print(f">>> KB direk kafasi (NW) ile S_left uyumu (Y/'-' = yanip sonen ok): {(nw == sl).mean():.1%} "
          f"-> NW, guneyden gelen sola donusun ikinci kafasi")

    # kontrol gorseli: 5 s arayla her grup + otomatik etiket
    tiles_by_grp = {g: [] for g in CROPS}
    for k in sorted(sheet_frames):
        fr = sheet_frames[k]
        row = sm.iloc[np.argmin(np.abs(sm.index - k))]
        for g, (x0, y0, x1, y1) in CROPS.items():
            c = cv2.resize(fr[y0:y1, x0:x1], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
            labs = [h for h in HEADS if (h.startswith(g) if g != "NW" else h == "NW")]
            for h in labs:
                for col, x, y in HEADS[h]:
                    cc = {"R": (0, 0, 255), "Y": (0, 220, 255), "F": (0, 160, 255), "G": (0, 255, 0)}[col]
                    cv2.circle(c, (int((x - x0) * 3), int((y - y0) * 3)), 5, cc, 1)
            txt = f"{k}s " + " ".join(f"{h}:{row[h]}" for h in labs)
            cv2.rectangle(c, (0, 0), (c.shape[1], 16), (0, 0, 0), -1)
            cv2.putText(c, txt, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            tiles_by_grp[g].append(c)
    blocks = []
    for g, tiles in tiles_by_grp.items():
        per = {"W": 6, "S": 4, "NW": 12}[g]
        rows = []
        for i in range(0, len(tiles), per):
            r = tiles[i:i + per]
            r += [np.zeros_like(tiles[0])] * (per - len(r))
            rows.append(np.hstack(r))
        blocks.append(np.vstack(rows))
    wmax = max(b.shape[1] for b in blocks)
    sheet = np.vstack([cv2.copyMakeBorder(b, 0, 8, 0, wmax - b.shape[1], cv2.BORDER_CONSTANT) for b in blocks])
    cv2.imwrite(args.sheet, sheet)
    print(f">>> Kontrol gorseli: {args.sheet}")


if __name__ == "__main__":
    main()
