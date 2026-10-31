#!/usr/bin/env python3
"""
reference_frame.png uzerinde tikladiginiz noktalarin PIKSEL koordinatlarini
terminale yazdiran basit bir yardimci arac. Homografi icin gereken
video-tarafi noktalari toplamak icin kullanin.
"""
import argparse
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True)
    args = ap.parse_args()

    img = cv2.imread(args.frame)
    if img is None:
        raise SystemExit(f"HATA: görüntü açılamadı: {args.frame}")

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"{x},{y}")
            cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
            cv2.imshow("pick_pixel_point (q ile cik)", img)

    cv2.namedWindow("pick_pixel_point (q ile cik)")
    cv2.setMouseCallback("pick_pixel_point (q ile cik)", click)
    cv2.imshow("pick_pixel_point (q ile cik)", img)

    print("Bir noktaya tıklayın, koordinati burada görünecek. Çıkmak için 'q'.")
    while True:
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()