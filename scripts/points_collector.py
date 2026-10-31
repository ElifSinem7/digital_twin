#!/usr/bin/env python3
"""
pick_pixel_point.py + elle points.csv duzenlemenin BIRLESIK, daha kolay
hali. Referans karede tikladiginiz her nokta icin, terminalden gercek
(sumo-gui'den okudugunuz) koordinati girmenizi ister ve otomatik olarak
points.csv'ye ekler.
"""
import argparse

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    img = cv2.imread(args.frame)
    if img is None:
        raise SystemExit(f"HATA: görüntü açılamadı: {args.frame}")

    collected = []
    clicked = {"pt": None}

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["pt"] = (x, y)

    win = "points_collector (tiklayin, sonra terminale donun)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)

    print(__doc__)
    print(f">>> Şu ana kadar toplanan nokta sayısı: {len(collected)}")

    while True:
        cv2.imshow(win, img)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break

        if clicked["pt"] is not None:
            px, py = clicked["pt"]
            clicked["pt"] = None
            cv2.circle(img, (px, py), 5, (0, 0, 255), -1)
            cv2.putText(img, str(len(collected) + 1), (px + 8, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            print(f"\n>>> Piksel noktasi #{len(collected)+1}: ({px}, {py})")
            real_input = input("    Gerçek x,y (sumo-gui'den okuduğunuz, 'iptal' yazıp atlayabilirsiniz): ").strip()
            if real_input.lower() == "iptal":
                print("    Atlandı.")
                continue
            try:
                rx, ry = map(float, real_input.replace(" ", "").split(","))
            except ValueError:
                print("    HATA: 'x,y' formatında girmediniz, bu nokta atlandı.")
                continue

            collected.append((px, py, rx, ry))
            print(f"    Eklendi. Toplam nokta: {len(collected)}")

    cv2.destroyAllWindows()

    if not collected:
        print(">>> Hiç nokta toplanmadı, dosya yazılmadı.")
        return

    with open(args.out, "w") as f:
        for px, py, rx, ry in collected:
            f.write(f"{px},{py},{rx},{ry}\n")

    print(f"\n>>> {len(collected)} nokta kaydedildi: {args.out}")
    if len(collected) < 4:
        print(">>> UYARI: en az 4 nokta gerekiyor, daha fazla toplamalısınız.")


if __name__ == "__main__":
    main()