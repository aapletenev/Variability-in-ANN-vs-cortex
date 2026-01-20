import pandas as pd

df = pd.read_csv('microns_area_labels.csv')
print(f"Total rows: {len(df)}")
print(f"\nUnique sessions: {df['session'].unique()}")
print(f"\nSession 4, Scan 7:")
filtered = df[(df['session'] == 4) & (df['scan_idx'] == 7)]
print(f"  Rows: {len(filtered)}")
print(f"  Unique unit_ids: {filtered['unit_id'].nunique()}")
print(f"  Min unit_id: {filtered['unit_id'].min()}")
print(f"  Max unit_id: {filtered['unit_id'].max()}")
print(f"\n  First few rows:")
print(filtered.head(10))
print(f"\n  Brain area counts:")
print(filtered['brain_area'].value_counts())
