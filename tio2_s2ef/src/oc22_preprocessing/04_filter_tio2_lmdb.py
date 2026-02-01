"""
Step 4: Filter TiO2 systems from OC22 pre-computed LMDBs
"""

import pickle
import sys
from pathlib import Path

import lmdb
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from oc22_preprocessing.oc22_path_config import DataPath, OC22DataPath


def load_tio2_system_ids():
    """Load TiO2 system IDs from filtered metadata"""
    if not DataPath.TIO2_SYSTEM_IDS_FILE.exists():
        print("Error: TiO2 system IDs not found!")
        print(f"Expected at: {DataPath.TIO2_SYSTEM_IDS_FILE}")
        print("Run 02_filter_tio2_systems.py first")
        return None
    
    system_ids = set()
    with open(DataPath.TIO2_SYSTEM_IDS_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            system_ids.add(line)
    
    print(f"Loaded {len(system_ids):,} TiO2 system IDs")
    return system_ids


def filter_lmdb_by_system_ids(input_lmdb_path, output_lmdb_path, tio2_ids, split_name):
    """
    Filter an LMDB database to keep only TiO2 systems
    """
    
    print(f"\nFiltering: {split_name}")
    print(f"  Input:  {input_lmdb_path}")
    print(f"  Output: {output_lmdb_path}")
    
    env_src = lmdb.open(
        str(input_lmdb_path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    
    # Create output LMDB
    output_lmdb_path.parent.mkdir(parents=True, exist_ok=True)
    env_dst = lmdb.open(
        str(output_lmdb_path),
        map_size=1024 * 1024 * 1024 * 0.5,  # 500 MB
        subdir=False,
        meminit=False,
        map_async=True,
    )
    
    new_idx = 0
    
    # Get total count for progress bar
    with env_src.begin() as txn:
        total_count = txn.stat()['entries']
    
    with env_src.begin() as txn_src:
        cursor = txn_src.cursor()
        
        for key, value in tqdm(cursor, total=total_count, desc=f"  Processing"):
            if key == b'length':
                continue
            
            try:
                data = pickle.loads(value)
                
                # Get system ID
                sid = None
                if hasattr(data, 'sid'):
                    sid = str(data.sid)
                elif hasattr(data, 'system_id'):
                    sid = str(data.system_id)
                elif hasattr(data, 'metadata') and 'system_id' in data.metadata:
                    sid = str(data.metadata['system_id'])
                
                if sid is None:
                    continue
                
                # Check if this system is TiO2
                if sid in tio2_ids:
                    txn_dst = env_dst.begin(write=True)
                    txn_dst.put(
                        f"{new_idx}".encode('ascii'),
                        value
                    )
                    txn_dst.commit()
                    new_idx += 1
                    
            except Exception as e:
                continue
    
    # Save count in destination LMDB
    txn_dst = env_dst.begin(write=True)
    txn_dst.put(b'length', pickle.dumps(new_idx, protocol=-1))
    txn_dst.commit()
    
    env_dst.sync()
    env_dst.close()
    env_src.close()
    
    print(f"  TiO2 entries saved: {new_idx:,}")

def main():
    print("=" * 60)
    print("STEP 4: Filter TiO2 Systems from OC22 LMDBs")
    
    # Load TiO2 system IDs
    tio2_ids = load_tio2_system_ids()
    if tio2_ids is None:
        return
    
    # Get all splits to process
    splits = OC22DataPath.get_all_splits()
    
    print(f"\nSplits to process: {', '.join(splits)}")
    print(f"TiO2 system IDs to filter: {len(tio2_ids):,}")
    
    # Process each split
    for split in splits:
        input_dir = OC22DataPath.get_split_dir(split)
        output_dir = OC22DataPath.get_tio2_split_dir(split)
        
        lmdb_files = list(input_dir.glob("*.lmdb"))
        
        if not lmdb_files:
            print(f"No LMDB files found in {input_dir}")
            print("Run 03_download_oc22_lmdb.py first")
            continue
        
        print(f"Found {len(lmdb_files)} LMDB file(s)")
        
        # Process each LMDB file
        for lmdb_file in lmdb_files:
            output_file = output_dir / lmdb_file.name
            
            filter_lmdb_by_system_ids(
                lmdb_file,
                output_file,
                tio2_ids,
                lmdb_file.name
            )
    
    # Final summary
    print(f"\nFiltered LMDBs saved to: {DataPath.TIO2_DIR}")
    print("\nFiltering Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()