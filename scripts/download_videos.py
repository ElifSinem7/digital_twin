"""
116th Ave NE & NE 12th St kavsagi icin paylasilan saatlik video
dosyalarini Google Drive'dan indirir.
"""
import argparse
import os

import gdown

VIDEOS = [
    ("Bellevue_116th_NE12th__2017-09-10_19-08-25.mp4", "1TeDzDnHVIrAX0lEv2qPcQUO_7Xy2fFgM"),
    ("Bellevue_116th_NE12th__2017-09-10_20-09-12.mp4", "1SVw6ctkA8xfpDJ7_2ssxId2Go3NqmiQa"),
    ("Bellevue_116th_NE12th__2017-09-10_21-08-54.mp4", "1PkmUxovlch4zsCfweKW59hCzW9HdBMGk"),
    ("Bellevue_116th_NE12th__2017-09-10_22-08-50.mp4", "1NFfJrbLWTSG7k1Yz07oCzyqI99bZza5v"),
    ("Bellevue_116th_NE12th__2017-09-10_23-08-29.mp4", "1kf7kIu4o95kYBg5SLsEeLkRyZW_xe_kD"),
    ("Bellevue_116th_NE12th__2017-09-11_00-08-29.mp4", "1dUK_Vnwac8HnVVoUMedDEaGylaPfjDTs"),
    ("Bellevue_116th_NE12th__2017-09-11_01-08-29.mp4", "1_nbOJyulR7zdj66senyVTlT-Jfy2xNNn"),
    ("Bellevue_116th_NE12th__2017-09-11_02-08-32.mp4", "1XV6zMx_BcSON2oZYOwoipxcIO1y78dbr"),
    ("Bellevue_116th_NE12th__2017-09-11_03-08-30.mp4", "11ImEodYfwf5EkhVwETnpYhyxZpm7RvYY"),
    ("Bellevue_116th_NE12th__2017-09-11_04-08-30.mp4", "1bsyenxn_O3fyH5GOVbpDac2apHJo6cYM"),
    ("Bellevue_116th_NE12th__2017-09-11_05-08-39.mp4", "1Oajy3rXqG25dAbBCAPGouqpPHSjpoKYN"),
    ("Bellevue_116th_NE12th__2017-09-11_06-08-30.mp4", "1gyeiyEx3UqxEVjf8YvvWnx1PuzgpjQVT"),
    ("Bellevue_116th_NE12th__2017-09-11_07-08-32.mp4", "11iEaThdLlOHeIrO53eZUtlyNPu0Irg3X"),
    ("Bellevue_116th_NE12th__2017-09-11_08-08-50.mp4", "1Cs4mvpztrlfmHeAcsAueoM4HcD-zcqLP"),
    ("Bellevue_116th_NE12th__2017-09-11_09-08-31.mp4", "1eRujpQWU7Q5F9yfPMuaH4rcXga_l7VOn"),
    ("Bellevue_116th_NE12th__2017-09-11_11-08-33.mp4", "1ze-CefZz5lwA2cdhXVgVlfndyoPH5H6Q"),
    ("Bellevue_116th_NE12th__2017-09-11_12-08-33.mp4", "1c3lvwz7FfTnTEVyLaopPg6NjpJ2pAn7Q"),
    ("Bellevue_116th_NE12th__2017-09-11_14-08-35.mp4", "1sj4tjNTzYArlY1HoH8t6D90D0Oawm50o"),
    ("Bellevue_116th_NE12th__2017-09-11_15-08-36.mp4", "1WED0ALcIG8tZrYGnmwhZB9_vdhk2Eq9t"),
    ("Bellevue_116th_NE12th__2017-09-11_16-08-37.mp4", "1QFcCPPBHSXVXUzkSHrNaF7dZJvF3UhR8"),
    ("Bellevue_116th_NE12th__2017-09-11_17-08-39.mp4", "1Uo2AsIXVeAu-2QoUvk8c1OW65rKuBlIE"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="../videos")
    ap.add_argument("--only-hour", type=str, default=None,
                     help="Sadece dosya adında bu alt-dizeyi içeren videoyu indir. ")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for filename, file_id in VIDEOS:
        if args.only_hour and args.only_hour not in filename:
            continue
        out_path = os.path.join(args.out_dir, filename)
        if os.path.exists(out_path):
            print(f"[atlanıyor] zaten var: {filename}")
            continue
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"[indiriliyor] {filename}")
        gdown.download(url, out_path, quiet=False)

    print("Bitti.")


if __name__ == "__main__":
    main()