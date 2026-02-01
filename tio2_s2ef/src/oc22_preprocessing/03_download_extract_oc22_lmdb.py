"""
Step 3: Download OC22 S2EF-Total Pre-computed LMDBs
Downloads and extracts ready-to-use LMDB datasets
"""

import os
import sys
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from oc22_preprocessing.oc22_path_config import DataPath, DownloadURLs


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path):
    """Download file with progress bar"""
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Downloading") as t:
        urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)
    
    size_gb = output_path.stat().st_size / (1024**3)
    print(f"Downloaded: {size_gb:.2f} GB")


def extract_tar_gz(tar_path, extract_to):
    """Extract .tar.gz file with progress bar"""
    print(f"\nExtracting: {tar_path.name}")
    
    extract_to.mkdir(parents=True, exist_ok=True)
    
    with tarfile.open(tar_path, 'r:gz') as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting"):
            tar.extract(member, path=extract_to)
    
    print(f"Extracted to: {extract_to}")


def download_and_extract_oc22_lmdb():
    """
    Download and extract OC22 S2EF-Total pre-computed LMDBs
    """
    
    print("Download size: ~20 GB compressed")
    print("Extracted size: ~70 GB")
    
    response = input("\nContinue with download? (y/n): ").lower()
    if response != 'y':
        print("Download cancelled.")
        return
    
    # Get paths
    data_dir = DataPath.DATA_DIR
    tar_path = data_dir / "s2ef_total_train_val_test_lmdbs.tar.gz"
    url = DownloadURLs.OC22_LMDB_URL
    
    # Create data directory
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download
    if tar_path.exists():
        print(f"\n{tar_path.name} already exists.")
        size_gb = tar_path.stat().st_size / (1024**3)
        print(f"Current file size: {size_gb:.2f} GB")
        
        response = input("Re-download? (y/n): ").lower()
        if response == 'y':
            tar_path.unlink()
            download_file(url, tar_path)
    else:
        download_file(url, tar_path)
    
    extract_tar_gz(tar_path, data_dir)
    
    # Verify extraction
    expected_dirs = [
        DataPath.TRAIN_DIR,
        DataPath.VAL_ID_DIR,
        DataPath.VAL_OOD_DIR,
        DataPath.TEST_ID_DIR,
        DataPath.TEST_OOD_DIR,
    ]
    
    all_found = True
    for expected_dir in expected_dirs:
        if expected_dir.exists():
            lmdb_files = list(expected_dir.glob("*.lmdb"))
            print(f"  ✓ {expected_dir.relative_to(data_dir)}: {len(lmdb_files)} LMDB file(s)")
        else:
            print(f"  ✗ {expected_dir.relative_to(data_dir)}: NOT FOUND")
            all_found = False
    
    if not all_found:
        print("\nSome expected directories are missing!")
        return
    
    # Cleanup
    print()
    response = input("Delete tar.gz file to save space? (y/n): ").lower()
    if response == 'y':
        if tar_path.exists():
            tar_path.unlink()
            print(f"Deleted: {tar_path.name}")
    
    print(f"\nData location: {data_dir.absolute()}")


if __name__ == "__main__":
    print("=" * 60)
    print(f"STEP 3: Download & Extract OC22 Pre-computed LMDBs")
    
    download_and_extract_oc22_lmdb()

    print("\nDownload And Extraction Complete!")
    print(f"{'=' * 60}")