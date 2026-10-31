#!/usr/bin/env bash
# Ayna modu demosunu uretir:
#   sabit-nesne maskesi -> maskeli tespit+takip -> [esleme] -> iz hazirlama -> sinyaller -> kayit -> tablolar
# Kullanim: [GPU_PY=/yol/python] bash mirror/make_mirror_demo.sh [baslangic=399] [sure=90] [cikti=../demo/bellevue_side_by_side_v3.mp4]
set -e
cd "$(dirname "$0")"
START="${1:-399}"; DUR="${2:-90}"; OUT="${3:-../demo/bellevue_side_by_side_v3.mp4}"
export SUMO_HOME="${SUMO_HOME:-$HOME/.local/lib/python3.10/site-packages/sumo}"
# Tespit adimlari icin Python: varsayilan etkin Python; GPU'lu ayri bir ortam icin GPU_PY=/yol/python
GPU_PY="${GPU_PY:-python}"
PY="$GPU_PY"
if "$PY" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then DEV=0; else DEV=cpu; fi
echo ">>> Tespit icin Python: $PY (cihaz: $DEV)"
VIDEO=../videos/Bellevue_116th_NE12th__2017-09-10_19-08-25.mp4

# 1) sabit-nesne maskesi (tum video, ~1 kare/s) - yoksa uret
[ -f static_mask.png ] || "$PY" -u build_static_mask.py --video "$VIDEO" --device "$DEV"
# 2) maskeli tespit + takip (demo araligi +-15 s)
"$PY" -u detect_window.py --video "$VIDEO" --start 384 --end 504 --mask static_mask.png \
    --out ../tracks/window_384_504_boxes.csv --device "$DEV"

source ../scripts/venv/bin/activate
# 3) esleme sabit tutulur (kontrol noktalari degismedi); yoksa yeniden kestir
[ -f fisheye_mapping.json ] || python -u fit_mapping.py
python -u prepare_mirror.py
[ -f signal_timeline_384_504.csv ] || python -u signal_reader.py
python -u infer_signals.py
python -u draw_valid_region.py
python -u record_mirror_demo.py --start "$START" --duration "$DUR" --out "$OUT"
python -u compare_counts.py --start "$START" --end "$(python -c "print($START + $DUR)")"
