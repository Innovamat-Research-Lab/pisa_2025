#!/usr/bin/env bash
# Section 4 of the paper (scaling all possessions items) and the simulation, 2022-2025, mathematics,
# science and reading. Run from revision_2026/ after the main Python pipeline (steps 02-04 write
# data/interim/python/homepos_irt.parquet), with PISA_RAW_DIR pointing to the OECD SPSS files.
# Each calibration takes 10-60 minutes and several GB of memory; run a few in parallel at most.
set -euo pipefail
mkdir -p data/metodos output/tables/metodos output/logs
python code/100_items_base.py 2022 2025          # items base from the public-use files
python code/101_valida_oficial_2025.py           # gate 0: published 2025 parameters reproduce HOMEPOS 2025
CAL=(
  "--items comunes --modelo gpcm --pi 0 --calib concurrente"            # common items, multigroup implementation
  "--items todos --modelo gpcm --pi 1 --calib concurrente"              # all items, concurrent calibration (Table 3)
  "--items todos --modelo gpcm --pi 0 --calib concurrente"              # no release
  "--items todos --modelo gpcm --pi 1 --calib concurrente --umbral 0.15 --tag 22-25_todos_gpcm_pi1_concurrente_u15"
  "--items todos --modelo gpcm --pi 1 --calib concurrente --umbral 0.20 --tag 22-25_todos_gpcm_pi1_concurrente_u20"
  "--items todos --modelo gpcm --pi 1 --calib concurrente --umbral 0.30 --tag 22-25_todos_gpcm_pi1_concurrente_u30"
  "--items todos --modelo pcm --pi 1 --calib concurrente"               # partial credit model
  "--items todos --modelo gpcm --pi 1 --calib fpl2022"                  # fixed parameters of 2022
  "--items todos --modelo gpcm --pi 1 --calib separada"                 # separate calibration by cycle
  "--items sin:st256 --modelo gpcm --pi 1 --calib concurrente"          # without the book types
  "--items sin:nac25 --modelo gpcm --pi 1 --calib concurrente"          # without the 2025 national items
  "--items sin:st256+nac25 --modelo gpcm --pi 1 --calib concurrente"    # without both
  "--items sin:st256+nac25+solo22+propios --modelo gpcm --pi 1 --calib concurrente"   # common items only
)
TAGS=()
for a in "${CAL[@]}"; do
  python code/102_calibra.py $a | tee -a output/logs/102_calibra.log
done
TAGS=$(ls data/metodos/hp_22-25_*.parquet | sed 's#.*/hp_##; s#\.parquet##' | tr '\n' ' ')
for dom in math scie read; do python code/103_huecos.py --dom $dom --tags $TAGS; done   # Table 3 and text
python code/104_propiedades.py 2022,2025 math     # reliability, correlations and their decomposition
python code/105_dif_comunes.py                    # drift of the common items, item groups
for s in S1 S2 S3 S4; do python code/106_simulacion.py $s 50; done   # Table S9, 50 replications
python code/107_mirt_check.py                     # cross-check against mirt (needs R with mirt)
