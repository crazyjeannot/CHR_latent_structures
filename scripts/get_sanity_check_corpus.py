import os
import random
import shutil
import pandas as pd

def copy_random_files_with_copies(source_dir, dest_dir, df_WER, num_files=1000, num_copies=5):
    bad_files = set(df_WER["index_name"].tolist())  
    # Convert to set for faster lookup# Step 1: List all .txt files in the source directory
    all_txt_files = [f for f in os.listdir(source_dir) if f.endswith('.txt') and f not in bad_files]
    
    # Step 2: Randomly select 100 files (or less if not enough files are present)
    selected_files = random.sample(all_txt_files, min(num_files, len(all_txt_files)))

    # Step 3: Create destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Step 4: Copy each file to the new directory and create 5 copies
    for file_name in selected_files:
        # Define full file paths
        source_file_path = os.path.join(source_dir, file_name)
        base_name, extension = os.path.splitext(file_name)

        # Copy the original file and make 5 additional copies
        for i in range(num_copies):  # +1 to include the original file
            if i == 0:
                # Original file (no suffix)
                dest_file_name = file_name
            else:
                # Copy files with _1, _2, _3, etc. suffixes
                dest_file_name = f"{base_name}_{i}{extension}"

            dest_file_path = os.path.join(dest_dir, dest_file_name)
            
            # Copy the file
            shutil.copy2(source_file_path, dest_file_path)

# Usage:
df_BAD_ocr = pd.read_csv('/data/jbarre/WER_GALLICA/df_BAD.csv') # sub 95% word error rate
source_directory = '/data/jbarre/fictions_gallica_txt'
destination_directory = '/data/jbarre/selected_gallica_1000'

copy_random_files_with_copies(source_directory, destination_directory, df_BAD_ocr)
