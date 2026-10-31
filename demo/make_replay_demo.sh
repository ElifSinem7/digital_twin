#!/usr/bin/env bash
# Birebir iz tekrari demo videosunu (demo/bellevue_side_by_side_replay.mp4) uretir:
# SUMO'daki her arac, videodaki gercek aracin konumuna her 0.1 s'de yerlestirilir.
# Kullanim:  bash demo/make_replay_demo.sh [baslangic_sn|auto] [sure_sn]
set -e
cd "$(dirname "$0")"

START="${1:-auto}"
DURATION="${2:-90}"

export SUMO_HOME="${SUMO_HOME:-$HOME/.local/lib/python3.10/site-packages/sumo}"
source ../scripts/venv/bin/activate

python -u record_replay_demo.py \
    --video ../videos/Bellevue_116th_NE12th__2017-09-10_19-08-25.mp4 \
    --tracks ../tracks/2017-09-10_19_stitched_real.csv \
    --out bellevue_side_by_side_replay.mp4 \
    --start "$START" --duration "$DURATION"
