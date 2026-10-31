#!/usr/bin/env bash
# Demo videosunu (demo/bellevue_side_by_side.mp4) yeniden uretir.
# Kullanim:  bash demo/make_demo.sh [baslangic_sn|auto] [sure_sn]
#   baslangic verilmezse (auto): ekranda en cok aracin oldugu aralik otomatik secilir.
set -e
cd "$(dirname "$0")"

START="${1:-auto}"
DURATION="${2:-90}"

export SUMO_HOME="${SUMO_HOME:-$HOME/.local/lib/python3.10/site-packages/sumo}"
source ../scripts/venv/bin/activate

# Orijinal rotalarin demo kopyasi: araclar kavsaga yakin ve yol hizinda baslar
python make_demo_routes.py --approach-m 40 --min-speed 8

if [ "$START" = "auto" ]; then
    # traci baglanirken stdout'a "Retrying..." yazabiliyor - sadece son satir (sayi) alinir
    START=$(python find_busy_window.py --sumocfg digital_twin_demo.sumocfg --duration "$DURATION" | tail -n 1)
fi
echo ">>> Kayit baslangici: ${START}s, sure: ${DURATION}s"

python -u record_demo.py \
    --video ../videos/Bellevue_116th_NE12th__2017-09-10_19-08-25.mp4 \
    --sumocfg digital_twin_demo.sumocfg \
    --out bellevue_side_by_side.mp4 \
    --start "$START" --duration "$DURATION"
