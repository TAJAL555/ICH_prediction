import pydicom
import numpy as np
import pandas as pd
import os
import logging
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    filename='preprocessing_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def process_dcm(args):
    dcm_file, dcm_folder, output_folder = args
    output_path = os.path.join(output_folder, dcm_file.replace('.dcm', '.npy'))
    if os.path.exists(output_path):
        return dcm_file  # Skip if already processed
    try:
        ds = pydicom.dcmread(os.path.join(dcm_folder, dcm_file))
        img = ds.pixel_array.astype(np.float32)
        img = np.clip(img, -1000, 1000)  # Clip Hounsfield Units
        img = (img + 1000) / 2000        # Normalize to [0, 1]
        np.save(output_path, img)
        return dcm_file
    except Exception as e:
        logging.error(f"Error processing {dcm_file}: {e}")
        return None

def main():
    dcm_folder = '' #the file is too large 483GB of size and i cannot upload it with it.
    output_folder = 'Data/'
    csv_file = 'stage_2_train.csv'

    os.makedirs(output_folder, exist_ok=True)

    dcm_files = [f for f in os.listdir(dcm_folder) if f.endswith('.dcm')][:100000]
    args_list = [(f, dcm_folder, output_folder) for f in dcm_files]

    num_processes = min(cpu_count(), 8)

    print(f"⚙️ Starting processing with {num_processes} processes...")

    processed_files = []
    with Pool(processes=num_processes) as pool:
        for result in tqdm(pool.imap_unordered(process_dcm, args_list), total=len(args_list)):
            if result is not None:
                processed_files.append(result)

    # Generate sampled CSV based on processed DICOM IDs
    processed_ids = [f.split('.')[0] for f in processed_files]
    df = pd.read_csv(csv_file)
    df_sample = df[df['ID'].str.split('_').str[:2].str.join('_').isin(processed_ids)]
    df_sample.to_csv(os.path.join(output_folder, 'stage_2_train_sample.csv'), index=False)

    print(f"✅ Finished: {len(processed_files)} DICOM files processed and saved as .npy")
    print(f"📄 Sample CSV saved to {output_folder}/stage_2_train_sample.csv")

if __name__ == '__main__':
    main()
