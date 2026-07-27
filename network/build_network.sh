# 116th Ave NE & NE 12th St (Bellevue, WA) - SUMO ağı üretim scripti
# Kavşak merkezi: 47.622219, -122.185783  (King County Metro durak verisi)
set -e

OUT_DIR="$(dirname "$0")"
cd "$OUT_DIR"

# Kavşağı ~300-350m yarıçapla kapsayan bir kutu oluşturuyoruz
BBOX="47.6195,-122.1895,47.6250,-122.1820"
OSM_FILE="bellevue_116th_ne12th.osm.xml"
NET_FILE="bellevue_116th_ne12th.net.xml"
POLY_FILE="bellevue_116th_ne12th.poly.xml"

echo ">>> [1/3] Overpass API'den OSM verisi indiriliyor (bbox: $BBOX)"
curl -X POST -d "data=[out:xml][timeout:60];(way[highway](${BBOX});node(w);way[building](${BBOX});node(w);>;);out meta;" \
     https://overpass-api.de/api/interpreter -o "$OSM_FILE"

if [ ! -s "$OSM_FILE" ]; then
    echo "HATA: OSM verisi indirilemedi."
    exit 1
fi

echo ">>> [2/3] netconvert ile SUMO ağı oluşturuluyor"
netconvert --osm-files "$OSM_FILE" \
    --output-file "$NET_FILE" \
    --geometry.remove \
    --roundabouts.guess \
    --ramps.guess \
    --junctions.join \
    --junctions.corner-detail 5 \
    --tls.guess-signals \
    --tls.discard-simple \
    --tls.join \
    --tls.default-type actuated \
    --sidewalks.guess \
    --crossings.guess \
    --keep-edges.by-vclass passenger,bicycle,pedestrian \
    --remove-edges.isolated

echo ">>> [3/3] Bina/yol poligonları (harita görünümü için) üretiliyor"
polyconvert --net-file "$NET_FILE" \
    --osm-files "$OSM_FILE" \
    --type-file "$SUMO_HOME/data/typemap/osmPolyconvert.typ.xml" \
    -o "$POLY_FILE"

echo ""
echo "TAMAMLANDI."
echo "  Ağ dosyası     : $NET_FILE"
echo "  Poligon dosyası: $POLY_FILE"
echo ""
