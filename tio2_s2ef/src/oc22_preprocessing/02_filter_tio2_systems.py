"""
Step 2: Filter TiO2 systems from OC22 metadata
Identifies all systems with TiO2 stoichiometry (Ti:O ratio = 1:2)
"""

import pickle
import json
import re
from pathlib import Path
from collections import Counter
from fractions import Fraction
import sys
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from oc22_preprocessing.oc22_path_config import DataPath


def parse_formula(bulk_symbols):
    """
    Parse chemical formula and return element counts
    Example: "Ti12O24" -> {'Ti': 12, 'O': 24}
    """
    elements = {}
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, bulk_symbols)
    
    for element, count in matches:
        if element:
            count = int(count) if count else 1
            elements[element] = count
    
    return elements


def is_tio2(bulk_symbols):
    """
    Check if bulk_symbols represents TiO2 stoichiometry (Ti:O ratio = 1:2)
    """
    elements = parse_formula(bulk_symbols)
    
    # Check if it's only Ti and O (no other elements)
    if set(elements.keys()) != {'Ti', 'O'}:
        return False
    
    # Check Ti:O ratio = 1:2
    ti_count = elements.get('Ti', 0)
    o_count = elements.get('O', 0)
    
    if ti_count == 0 or o_count == 0:
        return False
    
    # Reduce the ratio
    ratio = Fraction(ti_count, o_count)
    
    # TiO2 should have Ti:O = 1:2
    return ratio == Fraction(1, 2)


def load_metadata():
    """Load OC22 metadata file"""
    
    if not DataPath.OC22_MAPPING_FILE.exists():
        print("Error: Metadata file not found!")
        print(f"Expected at: {DataPath.OC22_MAPPING_FILE}")
        print("Run 01_download_oc22_metadata.py first")
        return None
    
    print(f"\nLoading metadata from: {DataPath.OC22_MAPPING_FILE}")
    with open(DataPath.OC22_MAPPING_FILE, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded {len(data):,} total systems")
    return data


def save_full_data_mapping(metadata):
    """
    Save full metadata as JSON for easy inspection
    Converts pickle format to JSON format
    """
    print("\nSaving full data mapping to JSON...")
    
    # Ensure metadata directory exists
    DataPath.METADATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-compatible format
    json_data = {}
    for system_id, info in tqdm(metadata.items(), desc="Converting to JSON"):
        json_data[str(system_id)] = {}
        for key, value in info.items():
            # Convert tuples to lists for JSON compatibility
            if isinstance(value, tuple):
                json_data[str(system_id)][key] = list(value)
            else:
                json_data[str(system_id)][key] = value
    
    # Save full mapping
    full_mapping_file = DataPath.METADATA_DIR / "full_data_mapping.json"
    with open(full_mapping_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Saved full data mapping to: {full_mapping_file}")
    print(f"Total systems: {len(json_data):,}")
    
    return full_mapping_file


def filter_tio2_systems(metadata):
    """Filter systems with TiO2 stoichiometry (Ti:O ratio = 1:2)"""
    print("\nFiltering TiO2 systems (Ti:O ratio = 1:2)")
    
    tio2_metadata = {}
    tio2_slab_only = {}
    tio2_adslab = {}
    
    for system_id, info in tqdm(metadata.items(), desc="Filtering"):
        bulk_symbols = info.get('bulk_symbols', '')
        
        # Check if it's TiO2 with correct stoichiometry
        if is_tio2(bulk_symbols):
            tio2_metadata[system_id] = info
            
            if 'ads_symbols' in info and info.get('nads', 0) > 0:
                tio2_adslab[system_id] = info
            else:
                tio2_slab_only[system_id] = info
    
    return tio2_metadata, tio2_slab_only, tio2_adslab


def analyze_tio2_systems(tio2_metadata, tio2_slab_only, tio2_adslab):
    """Analyze filtered TiO2 systems"""
    
    # Collect statistics
    adsorbates = Counter()
    bulks = Counter()
    miller_indices = Counter()
    
    for info in tio2_metadata.values():
        ads = info.get('ads_symbols', 'N/A')
        bulk = info.get('bulk_symbols', 'N/A')
        miller = str(info.get('miller_index', (0, 0, 0)))
        
        if ads != 'N/A':
            adsorbates[ads] += 1
        bulks[bulk] += 1
        miller_indices[miller] += 1
    
    # Prepare statistics
    stats = {
        'total_systems': len(tio2_metadata),
        'clean_slabs': len(tio2_slab_only),
        'with_adsorbates': len(tio2_adslab),
        'unique_adsorbates': len(adsorbates),
        'unique_bulks': len(bulks),
        'unique_miller': len(miller_indices),
        'top_adsorbates': dict(adsorbates.most_common(20)),
        'top_bulks': dict(bulks.most_common(20)),
        'top_miller': dict(miller_indices.most_common(20))
    }
    
    return stats


def save_tio2_results(tio2_metadata, stats):
    """Save filtered TiO2 data and statistics"""
    DataPath.METADATA_DIR.mkdir(parents=True, exist_ok=True)
    
    json_data = {}
    for system_id, info in tqdm(tio2_metadata.items(), desc="Converting TiO2 to JSON"):
        json_data[str(system_id)] = {}
        for key, value in info.items():
            if isinstance(value, tuple):
                json_data[str(system_id)][key] = list(value)
            else:
                json_data[str(system_id)][key] = value
    
    # Save TiO2 filtered mapping as JSON
    with open(DataPath.TIO2_MAPPING_FILE, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"\nSaved TiO2 mapping to: {DataPath.TIO2_MAPPING_FILE}")
    
    # Save system IDs
    with open(DataPath.TIO2_SYSTEM_IDS_FILE, 'w') as f:
        for sid in sorted(tio2_metadata.keys()):
            f.write(f"{sid}\n")
    print(f"Saved {len(tio2_metadata):,} system IDs to: {DataPath.TIO2_SYSTEM_IDS_FILE}")
    
    # Save statistics as JSON
    with open(DataPath.TIO2_STATISTICS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to: {DataPath.TIO2_STATISTICS_FILE}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("STEP 2: TiO2 System Filtering (OC22)")
    
    metadata = load_metadata()
    if metadata is None:
        sys.exit(1)
    
    full_mapping_file = save_full_data_mapping(metadata)
    
    tio2_metadata, tio2_slab_only, tio2_adslab = filter_tio2_systems(metadata)
    
    stats = analyze_tio2_systems(tio2_metadata, tio2_slab_only, tio2_adslab)
    
    save_tio2_results(tio2_metadata, stats)

    print("\nFinished TiO2 System Filtering")
    print(f"{'=' * 60}")