"""
Run all preprocessing steps in sequence
"""

import subprocess
import sys
from pathlib import Path


def run_script(script_name):
    """Run a preprocessing script"""
    script_path = Path(__file__).parent / script_name
    
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print('='*60)
    
    try:
        subprocess.run([sys.executable, str(script_path)], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError running {script_name}: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n\nInterrupted by user")
        return False


def main():
    print("="*60)
    print("TiO2 S2EF - Data Preprocessing Pipeline")
    print("="*60)
    
    scripts = [
        "01_download_metadata.py",
        "02_filter_tio2_systems.py",
        "03_download_s2ef_data.py",
        "04_create_tio2_lmdb.py",
    ]
    
    print(f"\nThis will run {len(scripts)} preprocessing steps:")
    for i, script in enumerate(scripts, 1):
        print(f"  {i}. {script}")
    
    user_input = input("\nContinue? (y/n): ").lower()
    if user_input != 'y':
        print("Cancelled.")
        return
    
    # Run each script
    for script in scripts:
        success = run_script(script)
        if not success:
            print(f"\nPipeline stopped at {script}")
            return
    
    print("ALL PREPROCESSING COMPLETE!")

if __name__ == "__main__":
    main()