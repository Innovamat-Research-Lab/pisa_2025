# Revision, November 2026

Code and result tables added for the journal version of the paper (*Did advantaged students really decline
most in PISA 2025? The socioeconomic index and the comparability of achievement gaps across cycles*). The
main pipeline in `code/python/` and `code/r/` is unchanged; this folder adds the analyses that the
revision introduced.

| Paper | Script | Output |
|---|---|---|
| Section 4 and Table 3 (scaling all possessions items: concurrent calibration with partial invariance, partial credit model, fixed parameters of 2022, separate calibration by cycle, release thresholds, removal of item blocks) | `100_items_base.py`, `102_calibra.py`, `103_huecos.py` | `output/tables/metodos/huecos_22-25_{math,scie,read}.csv` (rows `22-25_<items>_<model>_pi<0/1>_<calibration>`) |
| Section 4, reliability, correlation with achievement and its decomposition | `104_propiedades.py` | `output/tables/metodos/propiedades_22-25_math.csv` |
| Section 3.3 and Section 4, item groups and drift of the common items | `105_dif_comunes.py` | `output/tables/metodos/dif_comunes*.csv`, `items_especificos.csv` |
| Section 4, simulation (Table S9 of the supplement) | `106_simulacion.py` | `output/tables/metodos/sim_S{1..4}.csv` |
| Validation: the published 2025 item parameters reproduce the published 2025 HOMEPOS | `101_valida_oficial_2025.py` | `output/tables/metodos/101_valida_oficial_2025.csv` |
| Validation against the `mirt` package in R | `107_mirt_check.py` | `output/tables/metodos/107_mirt_check.csv` |
| Section 2.5 and Table S5, four-cycle series by comparability level of the items (13, 8 and 4 items) | `R1_four_cycle_item_tiers.py` | `output/tables/R1_four_cycle_item_tiers.csv` |
| Section 2.5, the 2022 filter on the device counts | `R2_device_filter_2022.py` | `output/tables/R2_device_filter_2022.csv` |
| Comparability level of each common item (wording, categories and filters in each cycle) | — | `output/tables/metodos/auditoria_items.csv` |

`_gpcm_mg.py` is the multigroup GPCM/PCM engine with partial invariance (release of item parameters by
country-by-language-by-cycle group when RMSD exceeds the threshold), `_items_homepos.py` the catalogue
and recoding of the possessions items in each cycle, `_escs.py` the OECD composite rule, `_brr.py` the
quartile gaps with the 80 BRR replicates and ten plausible values, and `_load.py` the reader of the
OECD files. Comments in these scripts are in Spanish.

## Running it

Run the main Python pipeline first (steps 02–04 write `data/interim/python/homepos_irt.parquet`, the
common-item index that also fixes the student sample). Then, from this folder:

```bash
cd revision_2026
export PISA_RAW_DIR=/path/to/oecd/spss      # CY08MSP_STU_QQQ.SAV and CY09_MS_STU_PUF.sav
bash code/run_section4.sh
```

`100_items_base.py` reads the possessions items, national items, language of the questionnaire and
published indices from the 2022 and 2025 public-use student files; the replicate weights come from the
repository extract (`data/extract`, Git LFS). Each calibration takes between ten minutes and an hour and
several gigabytes of memory. `107_mirt_check.py` needs R with the `mirt` package. The checks R1 and R2 run
on the repository extract with the main pipeline's library, as described in their headers.

The tables in `output/` are the ones reported in the paper.
