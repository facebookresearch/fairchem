"""
Step 2: Filter TiO2 systems from OC20 metadata
Identifies all systems containing titanium and oxygen in the bulk
"""

import pickle
from pathlib import Path
from collections import Counter
import sys
from tqdm import tqdm
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from oc20_preprocessing.oc20_path_config import DataPath

def load_metadata():
    """Load OC20 metadata file"""
    
    if not DataPath.OC20_MAPPING_FILE.exists():
        print("Error: Metadata file not found!")
        print(f"Expected at: {DataPath.OC20_MAPPING_FILE}")
        print("Run 01_download_metadata.py first")
        return None
    
    print(f"\nLoading metadata from: {DataPath.OC20_MAPPING_FILE}")
    with open(DataPath.OC20_MAPPING_FILE, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded {len(data):,} total systems")
    return data


def filter_tio2_systems(metadata):
    """Filter systems containing Ti and O in bulk"""
    print("\nFiltering TiO2 systems...")
    
    tio2_systems = {}
    
    for system_id, info in tqdm(metadata.items(), desc="Filtering"):
        bulk = info.get('bulk_symbols', '')
        
        # Check if both Ti and O are present
        if 'Ti' in bulk and 'O' in bulk:
            tio2_systems[system_id] = info
    
    print(f"\nFound {len(tio2_systems):,} TiO2 systems")
    print(f"  ({len(tio2_systems)/len(metadata)*100:.2f}% of total)")
    
    return tio2_systems


def analyze_tio2_systems(tio2_systems):
    """Analyze filtered TiO2 systems"""
    # Collect statistics
    adsorbates = Counter()
    bulks = Counter()
    miller_indices = Counter()
    
    for info in tio2_systems.values():
        ads = info.get('ads_symbols', 'unknown')
        bulk = info.get('bulk_symbols', 'unknown')
        miller = str(info.get('miller_index', (0, 0, 0)))
        
        adsorbates[ads] += 1
        bulks[bulk] += 1
        miller_indices[miller] += 1

    # Return stats as dictionary
    return {
        'total_systems': len(tio2_systems),
        'unique_adsorbates': len(adsorbates),
        'unique_bulks': len(bulks),
        'unique_miller': len(miller_indices),
        'top_adsorbates': dict(adsorbates.most_common(20)),
        'top_bulks': dict(bulks.most_common(20)),
        'top_miller': dict(miller_indices.most_common(20))
    }


def save_results(tio2_systems, stats):
    """Save filtered data and statistics"""
    
    # Ensure metadata directory exists
    DataPath.METADATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save filtered mapping
    with open(DataPath.TIO2_MAPPING_FILE, 'wb') as f:
        pickle.dump(tio2_systems, f)
    print(f"\nSaved TiO2 mapping to: {DataPath.TIO2_MAPPING_FILE}")
    
    # Save system IDs (one per line)
    with open(DataPath.TIO2_SYSTEM_IDS_FILE, 'w') as f:
        for sid in sorted(tio2_systems.keys()):
            f.write(f"{sid}\n")
    print(f"Saved {len(tio2_systems):,} system IDs to: {DataPath.TIO2_SYSTEM_IDS_FILE}")
    
    # Save statistics as JSON
    with open(DataPath.TIO2_STATISTICS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to: {DataPath.TIO2_STATISTICS_FILE}")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("STEP 2:TiO2 System Analysis")
    
    # Load metadata
    metadata = load_metadata()
    
    # Filter TiO2 systems
    tio2_systems = filter_tio2_systems(metadata)
    
    # Analyze
    stats = analyze_tio2_systems(tio2_systems)
    
    # Save results
    save_results(tio2_systems, stats)
    
    print("\nFinishing TiO2 System Filtering")
    print(f"{'=' * 60}")