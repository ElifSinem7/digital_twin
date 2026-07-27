"""
QGIS'i kullanıp uydu görüntüsü indirmeye çalıştım ama sürekli yanlış görüntüyü indirdiği için bu yönteme geçtim. Doğrudan Esri World Imagery sunucusunun
"export" API'sinden, net.xml'in tam koordinat sisteminde (UTM Zone 10N,
EPSG:32610) ve tam sınırlarında (convBoundary) georeferanslı bir uydu
görüntüsü indiricek.
"""
import argparse
import sys
import urllib.request
import urllib.parse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xmin", type=float, required=True)
    ap.add_argument("--ymin", type=float, required=True)
    ap.add_argument("--xmax", type=float, required=True)
    ap.add_argument("--ymax", type=float, required=True)
    ap.add_argument("--epsg", type=int, default=32610,
                     help="net.xml'deki projeksiyonla aynı EPSG kodu (UTM Zone 10N = 32610)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1200,
                     help="Çıktı görüntüsünün piksel genişliği")
    args = ap.parse_args()

    real_w = args.xmax - args.xmin
    real_h = args.ymax - args.ymin
    height = round(args.width * real_h / real_w)

    base_url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
    params = {
        "bbox": f"{args.xmin},{args.ymin},{args.xmax},{args.ymax}",
        "bboxSR": args.epsg,
        "imageSR": args.epsg,
        "size": f"{args.width},{height}",
        "format": "png32",
        "transparent": "false",
        "f": "image",
    }
    url = base_url + "?" + urllib.parse.urlencode(params)

    print(f">>> İstek gönderiliyor: {url}")
    print(f">>> Beklenen boyut: {args.width}x{height} px")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
    except Exception as e:
        sys.exit(f"HATA: indirme başarısız: {e}")

    if len(data) < 1000:
        sys.exit(f"HATA: çok küçük yanıt ({len(data)} byte) - sunucu hata mesajı dönmüş olabilir:\n{data[:500]}")

    with open(args.out, "wb") as f:
        f.write(data)

    print(f">>> Kaydedildi: {args.out} ({len(data)} byte)")

    # Dogrulama kısmı, gerçek bir PNG olup olmadığına bakıyoruz
    try:
        from PIL import Image
        img = Image.open(args.out)
        img.load()
        print(f">>> Doğrulandı: geçerli PNG, boyut={img.size}")
    except ImportError:
        print(">>> (Pillow kurulu degil, PNG doğrulaması atlandı - dosya muhtemelen sağlam)")
    except Exception as e:
        sys.exit(f"HATA: indirilen dosya bozuk görünüyor: {e}")


if __name__ == "__main__":
    main()