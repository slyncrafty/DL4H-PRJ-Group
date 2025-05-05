import os
import random
import subprocess
import pandas as pd
import getpass

def download_mimic_sample(
    metadata_csv,
    sample_ratio=0.05,
    username=None,
    destination="data/images/files",
    base_url="https://physionet.org/files/mimic-cxr-jpg/2.0.0/files/",
    seed=42
):
    # Load metadata
    column_names = ['dicom_id', 'subject_id', 'study_id', 'view', 'count']
    df = pd.read_csv(metadata_csv, header=None, names=column_names, delimiter="\t" if "\t" in open(metadata_csv).readline() else ",")

    # Get unique subject IDs
    unique_subject_ids = df['subject_id'].unique()
    print(f"📋 Total unique subjects in metadata: {len(unique_subject_ids)}")

    # Sample subject IDs
    sample_count = max(1, int(len(unique_subject_ids) * sample_ratio))
    random.seed(seed)
    sampled_subjects = random.sample(list(unique_subject_ids), sample_count)
    print(f"🎯 Selected {sample_count} subjects ({sample_ratio*100}%): seed {seed}")

    # Ask for PhysioNet password if not provided
    if username is None:
        username = input("Enter your PhysioNet username: ")
    password = getpass.getpass("Enter your PhysioNet password: ")

    # Prepare destination folder
    os.makedirs(destination, exist_ok=True)

    # Build URLs and download each subject folder
    for subj_id in sampled_subjects:
        subj_str = f"p{subj_id}"
        group_folder = subj_str[:3]  # Example: 'p10' from 'p10000032'
        full_url = os.path.join(base_url, group_folder, subj_str) + "/"
        print(f"⬇️ Downloading subject: {subj_str} from {full_url}")

        cmd = [
            "wget", "-r", "-N", "-c", "-np",
            "--user", username,
            "--password", password,
            full_url,
            "-P", destination
        ]

        subprocess.run(cmd)
    
    print("\n✅ Download completed for sampled subjects.")

# Example usage
if __name__ == "__main__":
    download_mimic_sample(
        metadata_csv="metadata/mimiccxr_test_sub_final.csv",
        sample_ratio=0.05,
        username=None, 
        destination="/sampled_mimic_download",
        seed=0
    )