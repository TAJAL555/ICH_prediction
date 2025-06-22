import pandas as pd
import os
from pathlib import Path

# Paths
npy_dir = "" #the file is too large 100GB of size and i cannot upload it with it.
rsna_csv = "stage_2_train.csv"
output_csv = "labels_filtered.csv"

# Get .npy filenames (without .npy extension)
npy_files = [f.stem for f in Path(npy_dir).glob("*.npy")]
print(f"Found {len(npy_files)} .npy files")
print("Sample .npy filenames:", npy_files[:5])

# Load RSNA labels
df = pd.read_csv(rsna_csv)
print("Sample stage_2_train.csv rows:")
print(df.head())

# Extract image_id (e.g., ID_000012eaf from ID_000012eaf_a)
df['image_id'] = df['ID'].str.extract(r'(ID_\w+)_')[0]  # Match ID_xxxxxx
print("Sample image_ids:", df['image_id'].unique()[:5])

# Aggregate labels (any ICH = 1)
df_agg = df.groupby('image_id')['Label'].max().reset_index()
df_agg.columns = ['image_id', 'ICH_label']
print(f"Aggregated to {len(df_agg)} unique image_ids")

# Filter for .npy files
df_filtered = df_agg[df_agg['image_id'].isin(npy_files)]
print(f"Filtered to {len(df_filtered)} matching labels")
print("Sample matching image_ids:", df_filtered['image_id'].head().tolist())

# Save new labels.csv
if len(df_filtered) == 0:
    print("Error: No matching image_ids. Check .npy filenames vs. stage_2_train.csv.")
else:
    df_filtered.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")