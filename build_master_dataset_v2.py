"""
build_master_dataset_v2.py
==========================
Builds the CORRECTED Habulator master training dataset from three sources:

  1. GLENDA_with_TP_TEMPS_SIS_PHYTO.csv
       Main phytoplankton biovolume + water chemistry (TP, TEMP, SI, CHLA).
       Biovolume columns (EDIAT, LDIAT, CHLOR, CRYPT, CYANO, Total) are
       species-specific values from U.S. EPA GLNPO (µm³/mL → mg/L).

  2. output1.csv  (2001–2019)
       NO23 (nitrate+nitrite, mg/L) — from coauthor.

  3. Water Quality Data 2021 v1.xlsx  (2021 only)
       NO23 for 2021 stations from U.S. EPA GLNPO annual release.

Changes from v1 (build_master_dataset.py):
──────────────────────────────────────────
  1. REMOVED QC replicates: field duplicates (SAMPLE_ID 7th char = 'D')
     and lab duplicates (7th char = 'C') are filtered out. These exist for
     inter-analyst precision checks per EPA SOP LG401 §9.0, not as
     independent ecological observations.

  2. REMOVED join-artifact duplicates: The GLENDA CSV contains 657
     station+date pairs where the same chemistry record was joined to
     two separate phytoplankton biovolume records (same SAMPLE_ID,
     different biovolume values). These arise from the EPA's two-part
     counting protocol (soft-algae Utermöhl count + cleaned diatom slide
     count, per SOP LG401 §6.4.1). We keep only the first occurrence
     per station+date (drop_duplicates, keep='first').

  3. NO IMPUTATION for NO23 — same as v1.

  4. Year 2020: No EPA sampling — excluded automatically.
     Year 2022: No NO23/TP/SI available — excluded.

References:
  - EPA SOP LG401 v07: Phytoplankton Analysis (Modified Utermöhl Method)
    https://www.epa.gov/system/files/documents/2021-12/lg401.v07-phytoplankton-analysis_rfa.pdf
  - EPA GLNPO Water Quality Data 2021 v1
    https://cdx.epa.gov/

Output: habulator_master_v2.csv
"""

import pandas as pd
import numpy as np

MAIN_PATH   = 'GLENDA_with_TP_TEMPS_SIS_PHYTO.csv'
NO23_PATH   = 'output1.csv'
WQ2021_PATH = 'Water Quality Data 2021 v1.xlsx'
OUT_PATH    = 'habulator_master_v2.csv'

print("=" * 70)
print("Habulator Master Dataset Builder  v2 (corrected — no duplicates)")
print("=" * 70)

# ── 1. Load sources ──────────────────────────────────────────────────────────
print("\n[1/8] Loading source files...")
main = pd.read_csv(MAIN_PATH)
no23_src = pd.read_csv(NO23_PATH)
wq2021 = pd.read_excel(WQ2021_PATH, sheet_name='integrated samples')

print(f"  GLENDA file       : {len(main):,} rows")
print(f"  NO23 file (out1)  : {len(no23_src):,} rows  (2001–2019)")
print(f"  WQ 2021 xlsx      : {len(wq2021):,} rows")

# ── 2. Remove QC replicates ──────────────────────────────────────────────────
print("\n[2/8] Removing QC replicates (field + lab duplicates)...")
print(f"  QC_TYPE distribution (before):")
print(f"    {main['QC_TYPE'].value_counts().to_dict()}")

n_before = len(main)
main = main[main['QC_TYPE'] == 'routine field sample'].copy()
n_removed_qc = n_before - len(main)

print(f"  Removed {n_removed_qc} QC replicate rows (field/lab duplicates)")
print(f"  Remaining: {len(main):,} routine field samples")
print(f"  Rationale: QC replicates are inter-analyst precision checks")
print(f"             per EPA SOP LG401 §9.0, not independent observations.")

# ── 3. Remove join-artifact duplicates ────────────────────────────────────────
print("\n[3/8] Removing join-artifact duplicates (same station+date)...")
main['SAMPLING_DATE'] = pd.to_datetime(main['SAMPLING_DATE'])
main['stn'] = main['station_norm'].str.strip().str.upper()
main['YEAR_int'] = main['SAMPLING_DATE'].dt.year

n_before_dedup = len(main)
n_dup_groups = main.duplicated(subset=['stn', 'SAMPLING_DATE'], keep='first').sum()

main = main.drop_duplicates(subset=['stn', 'SAMPLING_DATE'], keep='first').copy()
n_removed_dup = n_before_dedup - len(main)

print(f"  Duplicate station+date rows removed: {n_removed_dup}")
print(f"  Remaining: {len(main):,} unique station+date observations")
print(f"  Rationale: GLENDA CSV joined two phytoplankton biovolume records")
print(f"             (from EPA's two-part counting protocol, SOP LG401 §6.4.1)")
print(f"             to the same chemistry record. We keep the first occurrence.")

# ── 4. Normalise keys ────────────────────────────────────────────────────────
print("\n[4/8] Normalising join keys...")
no23_src['Date']     = pd.to_datetime(no23_src['Date'])
no23_src['YEAR_int'] = no23_src['Date'].dt.year
no23_src['stn']      = no23_src['StationID'].str.strip().str.upper()

wq2021['NO23_raw']   = pd.to_numeric(wq2021['NO2+NO3, mg/L'], errors='coerce')
wq2021['YEAR_int']   = 2021
wq2021['stn']        = (wq2021['STATION']
                        .str.strip().str.upper()
                        .str.replace(' ', '', regex=False)
                        .str.replace(r'(\d)M$', r'\1', regex=True))

# ── 5. Build NO23 lookup — DIRECT MATCH ONLY ─────────────────────────────────
print("\n[5/8] Building NO23 lookup (direct match only, no imputation)...")

t1_out1 = (no23_src
           .groupby(['stn', 'YEAR_int'])['NO23']
           .median()
           .reset_index())
print(f"  output1 lookup    : {len(t1_out1):,} station+year pairs  (2001–2019)")

t1_wq = (wq2021[wq2021['NO23_raw'].notna()]
         .groupby(['stn', 'YEAR_int'])['NO23_raw']
         .median()
         .reset_index()
         .rename(columns={'NO23_raw': 'NO23'}))
print(f"  WQ 2021 lookup    : {len(t1_wq):,} station+year pairs  (2021)")

no23_lookup = pd.concat([t1_out1, t1_wq], ignore_index=True)
print(f"  Combined lookup   : {len(no23_lookup):,} entries")

# ── 6. Drop 2022 and merge NO23 ──────────────────────────────────────────────
print("\n[6/8] Dropping year 2022 and merging NO23...")
n_before_2022 = len(main)
main = main[main['YEAR_int'] != 2022].copy()
print(f"  Dropped {n_before_2022 - len(main)} rows from 2022")

df = main.merge(no23_lookup, on=['stn', 'YEAR_int'], how='left')

n_matched   = df['NO23'].notna().sum()
n_unmatched = df['NO23'].isna().sum()
print(f"  NO23 matched   : {n_matched:,} rows  ({n_matched/len(df)*100:.1f}%)")
print(f"  NO23 unmatched : {n_unmatched:,} rows  ({n_unmatched/len(df)*100:.1f}%) → dropped")

df = df[df['NO23'].notna()].copy()
print(f"  After drop: {len(df):,} rows")

# ── 7. Validation ────────────────────────────────────────────────────────────
print("\n[7/8] Validating...")

no23_min = df['NO23'].min()
no23_max = df['NO23'].max()
assert df['NO23'].isna().sum() == 0,  "ERROR: NO23 NaN values remain"
assert no23_min >= 0,                 "ERROR: Negative NO23 value found"
assert no23_max < 5.0,                "ERROR: Implausibly high NO23 value found"
print(f"  NO23 range       : [{no23_min:.4f}, {no23_max:.4f}] mg/L  ✓")

# Confirm no duplicates remain
n_dup_final = df.duplicated(subset=['stn', 'SAMPLING_DATE']).sum()
assert n_dup_final == 0, f"ERROR: {n_dup_final} duplicate station+dates remain"
print(f"  Duplicate check  : 0 duplicates  ✓")

for feat in ['TEMP', 'TP', 'SI', 'STN_DEPTH_M']:
    n_na = df[feat].isna().sum() if feat in df.columns else -1
    print(f"  {feat:15s} NaN: {n_na}")

years = sorted(df['YEAR_int'].unique())
print(f"  Years: {years[0]}–{years[-1]}  ({len(years)} years)")
print(f"  No 2020 (no sampling): {'✓' if 2020 not in years else '⚠ WARNING'}")
print(f"  No 2022 (no NO23):     {'✓' if 2022 not in years else '⚠ WARNING'}")

# Year distribution
print(f"\n  Rows per year:")
for yr in years:
    n = len(df[df['YEAR_int'] == yr])
    print(f"    {yr}: {n:4d}")

# ── 8. Engineer features and save ────────────────────────────────────────────
print("\n[8/8] Engineering features and saving...")

df['DOY']      = df['SAMPLING_DATE'].dt.dayofyear
df['NP_ratio'] = df['NO23'] / df['TP'].clip(lower=0.001)

FINAL_COLS = [
    'YEAR', 'LAKE', 'station_norm', 'SAMPLING_DATE',
    'LATITUDE', 'LONGITUDE',
    'TEMP', 'TP', 'SI', 'NO23', 'STN_DEPTH_M',
    'DOY', 'NP_ratio',
    'EDIAT', 'LDIAT', 'CHLOR', 'CRYPT', 'CYANO', 'Total',
]

df_out = df[FINAL_COLS].copy()
df_out.to_csv(OUT_PATH, index=False)

print(f"\n{'='*70}")
print(f"CORRECTED MASTER DATASET SAVED: {OUT_PATH}")
print(f"{'='*70}")
print(f"  Total rows    : {len(df_out):,}")
print(f"  Total columns : {len(df_out.columns)}")
print(f"  Features      : TEMP, TP, SI, NO23, STN_DEPTH_M, DOY, NP_ratio")
print(f"  Targets       : EDIAT, LDIAT, CHLOR, CRYPT, CYANO, Total")
print(f"  Years         : {years[0]}–{years[-1]}  (no 2020, no 2022)")
print(f"")
print(f"  Data curation applied:")
print(f"    1. Removed {n_removed_qc} QC replicate rows (field/lab duplicates)")
print(f"    2. Removed {n_removed_dup} join-artifact duplicate rows")
print(f"       (same station+date, different biovolume from two-part count)")
print(f"    3. NO23 direct match only (no imputation)")
print(f"    4. Dropped rows without measured NO23")
print(f"")

# Final missing value report
print(f"Missing values per column:")
miss = df_out.isnull().sum()
miss_any = miss[miss > 0]
if len(miss_any) == 0:
    print("  None — all columns fully populated  ✓")
else:
    for col, n in miss_any.items():
        print(f"  {col:20s}: {n:4d}  ({n/len(df_out)*100:.1f}%)")

# Comparison with v1
print(f"\n{'─'*70}")
print(f"COMPARISON WITH v1 (habulator_master.csv):")
print(f"  v1 rows: 3,226  (with duplicates)")
print(f"  v2 rows: {len(df_out):,}  (clean, no duplicates)")
print(f"  Removed : {3226 - len(df_out):,} rows")
print(f"  All {len(df_out):,} rows are unique station+date observations")
print(f"{'─'*70}")

print("\nDone.")
