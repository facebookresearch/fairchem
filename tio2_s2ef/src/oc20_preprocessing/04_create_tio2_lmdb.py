"""
Step 4: Create TiO2-specific LMDB database from extxyz files
Processes extxyz and txt files, then filters to keep only TiO2 systems
"""

import glob
import multiprocessing as mp
import os
import pickle
import sys
from pathlib import Path

import ase.io
import lmdb
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from oc20_preprocessing.oc20_path_config import DataPath

# Import from ocpmodels at root level
# Structure: D:\fairchem-tio2-s2ef\ocpmodels\preprocessing\atoms_to_graphs.py
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_dir))

try:
    from ocpmodels.preprocessing.atoms_to_graphs import AtomsToGraphs
except ImportError as e:
    print(f"\nError: Could not import AtomsToGraphs")
    print(f"Searched in: {root_dir}")
    print(f"Error details: {e}")
    print(f"\nExpected file: {root_dir / 'ocpmodels' / 'preprocessing' / 'atoms_to_graphs.py'}")
    sys.exit(1)


def load_tio2_system_ids():
    """Load TiO2 system IDs"""
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
            
            # Handle different formats:
            # Format 1: "random1005013" -> extract 1005013
            # Format 2: "1005013" -> use directly
            if line.startswith('random'):
                sid = int(line.replace('random', ''))
            else:
                sid = int(line)
            
            system_ids.add(sid)
    
    print(f"Loaded {len(system_ids):,} TiO2 system IDs")
    return system_ids


def write_images_to_lmdb(mp_arg):
    """Process extxyz files and write to LMDB with TiO2 filtering"""
    a2g, db_path, samples, sampled_ids, idx, pid, tio2_ids = mp_arg
    
    db = lmdb.open(
        db_path,
        map_size=1024 * 1024 * 1024 * 10,  # 100 GB
        subdir=False,
        meminit=False,
        map_async=True,
    )

    pbar = tqdm(
        total=len(samples) * 100,  # Estimate
        position=pid,
        desc=f"Worker {pid}",
    )
    
    kept_count = 0
    skipped_count = 0
    
    for sample in samples:
        # Read trajectory log file
        try:
            traj_logs = open(sample, "r").read().splitlines()
        except:
            continue
        
        # Get corresponding extxyz file
        xyz_idx = os.path.splitext(os.path.basename(sample))[0]
        traj_path = os.path.join(os.path.dirname(sample), f"{xyz_idx}.extxyz")
        
        if not os.path.exists(traj_path):
            continue
        
        # Read trajectory frames
        try:
            traj_frames = ase.io.read(traj_path, ":")
        except:
            continue
        
        pbar.total = pbar.n + len(traj_frames)
        
        for i, frame in enumerate(traj_frames):
            if i >= len(traj_logs):
                break
            
            frame_log = traj_logs[i].split(",")
            
            # Extract system ID and frame ID
            try:
                sid = int(frame_log[0].split("random")[1])
                fid = int(frame_log[1].split("frame")[1])
            except:
                pbar.update(1)
                continue
            
            # TiO2 filtering - skip if not in TiO2 list
            if sid not in tio2_ids:
                skipped_count += 1
                pbar.update(1)
                continue
            
            # Convert to graph
            try:
                data_object = a2g.convert(frame)
            except:
                pbar.update(1)
                continue
            
            # Add metadata
            data_object.tags = torch.LongTensor(frame.get_tags())
            data_object.sid = sid
            data_object.fid = fid
            
            # Write to LMDB
            txn = db.begin(write=True)
            txn.put(
                f"{idx}".encode("ascii"),
                pickle.dumps(data_object, protocol=-1),
            )
            txn.commit()
            
            idx += 1
            kept_count += 1
            sampled_ids.append(",".join(frame_log[:2]) + "\n")
            pbar.update(1)

    # Save count
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn.commit()

    db.sync()
    db.close()
    pbar.close()

    return sampled_ids, idx, kept_count, skipped_count


def main():
    print("=" * 60)
    print("STEP 4: Create TiO2-Filtered LMDB")
    
    # Configuration
    NUM_WORKERS = 1 
    
    # Uncompressed S2EF data
    DATA_PATH = DataPath.UNCOMPRESSED_200K_DIR
    
    # Output goes to TiO2 LMDB directory
    OUT_PATH = DataPath.TIO2_LMDB_200K_DIR
    
    print(f"\nConfiguration:")
    print(f"  Data path: {DATA_PATH}")
    print(f"  Output path: {OUT_PATH}")
    print(f"  Workers: {NUM_WORKERS}")
    
    # Verify data path exists
    if not DATA_PATH.exists():
        print(f"\nError: Data path does not exist: {DATA_PATH}")
        print("Run 03_download_s2ef_data.py first to download and uncompress data")
        return
    
    # Load TiO2 system IDs for filtering
    tio2_ids = load_tio2_system_ids()
    if tio2_ids is None:
        return
    
    # Find txt files
    txt_files = glob.glob(str(DATA_PATH / "*.txt"))
    if not txt_files:
        print(f"\nError: No *.txt files found in {DATA_PATH}")
        return
    
    print(f"\nFound {len(txt_files)} trajectory log files")
    
    num_workers = min(NUM_WORKERS, len(txt_files))
    if num_workers != NUM_WORKERS:
        print(f"Adjusted workers to {num_workers} (number of txt files)")
    
    # Initialize feature extractor
    a2g = AtomsToGraphs(
        max_neigh=50,
        radius=6,
        r_energy=True,
        r_forces=True,
        r_fixed=True,
        r_distances=False,
        r_edges=False,
    )
    
    # Create output directory
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    
    # Initialize LMDB paths
    db_paths = [
        str(OUT_PATH / f"data.{i:04d}.lmdb")
        for i in range(num_workers)
    ]
    
    # Chunk files
    chunked_txt_files = np.array_split(txt_files, num_workers)
    
    # Process in parallel
    sampled_ids = [[]] * num_workers
    idx = [0] * num_workers
    
    print(f"\nProcessing with {num_workers} workers...")
    
    pool = mp.Pool(num_workers)
    mp_args = [
        (
            a2g,
            db_paths[i],
            chunked_txt_files[i],
            sampled_ids[i],
            idx[i],
            i,
            tio2_ids,
        )
        for i in range(num_workers)
    ]
    
    results = list(pool.imap(write_images_to_lmdb, mp_args))
    pool.close()
    pool.join()
    
    # Unpack results
    sampled_ids = [r[0] for r in results]
    idx = [r[1] for r in results]
    kept_counts = [r[2] for r in results]
    skipped_counts = [r[3] for r in results]
    
    # Save logs
    for j, i in enumerate(range(num_workers)):
        ids_log = open(
            str(OUT_PATH / f"data_log.{i:04d}.txt"), "w"
        )
        ids_log.writelines(sampled_ids[j])
        ids_log.close()
    
    # Print summary
    total_kept = sum(kept_counts)
    total_skipped = sum(skipped_counts)
    total_processed = total_kept + total_skipped
    

    print(f"\nTotal frames processed: {total_processed:,}")
    print(f"TiO2 frames kept: {total_kept:,}")
    print(f"Non-TiO2 frames skipped: {total_skipped:,}")
    print(f"Filtering rate: {(total_skipped/total_processed)*100:.1f}% filtered out")
    
    print(f"\nOutput location: {OUT_PATH}")

    print("PROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()