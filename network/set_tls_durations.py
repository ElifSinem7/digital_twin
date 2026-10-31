#!/usr/bin/env python3
"""
net.xml icine GOMULU tlLogic'in faz surelerini (duration) dogrudan
degistirir - additional-file eklemeye gerek kalmadan.

ONEMLI DUZELTME: Fazlari artik KONUMLARINA (dosyadaki sirasina) gore
degistiriyoruz, DEGERLERINE gore degil. Onceki versiyon, ayni sureye
sahip birden fazla faz oldugunda (ornegin iki farkli faz da
duration="3" ise) yanlislikla birbirine karisabiliyordu - artik boyle
bir riski yok, cunku her fazi dosyadaki POZISYONUYLA esliyoruz.
"""
import argparse
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--tls-id", required=True)
    ap.add_argument("--durations", nargs="+", type=float, required=True)
    args = ap.parse_args()

    with open(args.net, "r", encoding="utf-8") as f:
        content = f.read()

    escaped_id = re.escape(args.tls_id)
    block_pattern = re.compile(
        rf'(<tlLogic id="{escaped_id}"[^>]*>)(.*?)(</tlLogic>)',
        re.DOTALL,
    )
    match = block_pattern.search(content)
    if not match:
        raise SystemExit(f"HATA: tlLogic id bulunamadı: {args.tls_id}")

    header, body, footer = match.groups()

    phase_pattern = re.compile(r'<phase duration="[\d.]+"')
    n_phases = len(phase_pattern.findall(body))

    if n_phases != len(args.durations):
        raise SystemExit(
            f"HATA: {n_phases} faz var ama {len(args.durations)} süre verdiniz. "
            f"Sayılar eşleşmeli."
        )

    # KONUM BAZLI degistirme: re.sub'a bir "sayac" fonksiyonu veriyoruz,
    # her eslesmede SIRADAKI yeni degeri kullanip bir sonraki fazina geciyor.
    # Bu, ayni deger birden fazla fazda tekrarlansa bile HER ZAMAN doğru
    # POZISYONU degistirir - metin eslesmesi (string matching) DEGIL.
    counter = {"i": 0}

    def replacer(m):
        idx = counter["i"]
        counter["i"] += 1
        new_dur = args.durations[idx]
        return f'<phase duration="{new_dur:g}"'

    new_body = phase_pattern.sub(replacer, body)

    new_content = content[:match.start()] + header + new_body + footer + content[match.end():]

    with open(args.net, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f">>> {n_phases} faz süresi konum bazlı güncellendi: {args.durations}")
    print(f">>> Kaydedildi: {args.net}")


if __name__ == "__main__":
    main()