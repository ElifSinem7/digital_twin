"""
net.xml içine gömülü tlLogic'in faz sürelerini (duration) doğrudan değiştirir. 
10 değer, tlLogic içindeki 10 <phase>'e sırayla karşılık gelir:
  1. K-G yeşil (actuated, min/max korunur)
  2. K-G yeşil (min-durum-sonrasi sabit kisim)
  3. K-G sarı
  4. K-G sola-dönüş yeşil
  5. K-G sola-dönüş sarı
  6. D-B yeşil (actuated)
  7. D-B yeşil (min-durum-sonrası sabit kısım)
  8. D-B sarı
  9. D-B sola-dönüş yeşil
  10. D-B sola-dönüş sarı
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

    # tlLogic bloğunu bulma kısmı
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
    phases = phase_pattern.findall(body)

    if len(phases) != len(args.durations):
        raise SystemExit(
            f"HATA: {len(phases)} faz var ama {len(args.durations)} süre verdiniz. "
            f"Sayılar eşleşmeli."
        )

    new_body = body
    for old_dur, new_dur in zip(phases, args.durations):
        new_body = new_body.replace(
            old_dur, f'<phase duration="{new_dur:g}"', 1
        )

    new_content = content[:match.start()] + header + new_body + footer + content[match.end():]

    with open(args.net, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f">>> {len(phases)} faz süresi güncellendi: {args.durations}")
    print(f">>> Kaydedildi: {args.net}")


if __name__ == "__main__":
    main()