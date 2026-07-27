set -e

cd "$(dirname "$0")"

VIDEO_DIR="../videos"
TRACKS_DIR="../tracks"
CONFIG="../config.json"
ZONE_EDGE_MAP="../zone_to_edge.json"

mkdir -p "$VIDEO_DIR" "$TRACKS_DIR"

echo "=== [1/4] Videolar indiriliyor ==="
python download_videos.py --out-dir "$VIDEO_DIR"

if [ ! -f "$CONFIG" ]; then
    echo ""
    echo "HATA: $CONFIG bulunamadı."
    exit 1
fi

if [ ! -f "$ZONE_EDGE_MAP" ]; then
    echo ""
    echo "HATA: $ZONE_EDGE_MAP bulunamadı."
    exit 1
fi

echo ""
echo "=== [2/4] Her video için araç takibi (YOLO) ==="
for video in "$VIDEO_DIR"/*.mp4; do
    name=$(basename "$video" .mp4)
    out="$TRACKS_DIR/${name}.csv"
    if [ -f "$out" ]; then
        echo "[atlanıyor] zaten var: $out"
        continue
    fi
    python vehicle_tracker.py --video "$video" --out "$out" --sample-every 2
done

echo ""
echo "=== [3/4] Sinyal fazlarının çıkarılması (ilk video referans alınarak) ==="
FIRST_VIDEO=$(ls "$VIDEO_DIR"/*.mp4 | head -n 1)
python signal_light_extractor.py \
    --video "$FIRST_VIDEO" \
    --config "$CONFIG" \
    --out-timeline ../signal_timeline.csv \
    --out-tls ../sumo/tls_auto.add.xml \
    --tls-id TLSID \
    --sample-fps 2

echo ""
echo "=== [4/4] Saat saat gerçekçi talep (routes) üretimi ==="
python movement_to_routes.py \
    --tracks-dir "$TRACKS_DIR" \
    --config "$CONFIG" \
    --zone-edge-map "$ZONE_EDGE_MAP" \
    --out ../sumo/routes_auto.rou.xml

echo ""
echo "TAMAMLANDI."
echo "  ../sumo/tls_auto.add.xml    -> gerçek videodan ölçülmüş sinyal fazları"
echo "  ../sumo/routes_auto.rou.xml -> saat saat gerçek araç hacimleri"
echo ""