import pandas as pd
import sys

snpp_path = sys.argv[1]  # e.g. suomi_viirs_c2.csv
noaa20_path = sys.argv[2]  # e.g. j1_viirs_c2.csv
out_path = sys.argv[3] if len(sys.argv) > 3 else "bootleg_viirs_archive.csv"

df_snpp = pd.read_csv(snpp_path)
df_noaa20 = pd.read_csv(noaa20_path)

print(f"S-NPP: {len(df_snpp)} rows, columns: {list(df_snpp.columns)}")
print(f"NOAA-20: {len(df_noaa20)} rows, columns: {list(df_noaa20.columns)}")

# Tag provenance explicitly in case 'satellite' col is missing or ambiguous
if 'satellite' not in df_snpp.columns:
    df_snpp['satellite'] = 'SNPP'
if 'satellite' not in df_noaa20.columns:
    df_noaa20['satellite'] = 'NOAA20'

assert list(df_snpp.columns) == list(df_noaa20.columns), \
    "Column mismatch between the two CSVs -- check schema before merging."

df_combined = pd.concat([df_snpp, df_noaa20], ignore_index=True)
df_combined = df_combined.sort_values(['acq_date', 'acq_time'])

df_combined.to_csv(out_path, index=False)
print(f"Combined: {len(df_combined)} rows -> {out_path}")
print(df_combined['satellite'].value_counts())