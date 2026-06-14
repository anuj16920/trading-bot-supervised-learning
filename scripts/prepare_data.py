"""Prepare OHLCV data for AQRF training.

Copies data from EURUSD/ohlcv to data/EURUSD and processes it.
"""
import shutil
from pathlib import Path

def prepare_data():
    """Copy and organize data files."""
    source_dir = Path("EURUSD/ohlcv")
    
    # Map source to destination
    mappings = {
        "1min": "data/EURUSD/M1",
        "1hour": "data/EURUSD/H1",
    }
    
    for source_subdir, dest_dir in mappings.items():
        source_path = source_dir / source_subdir
        dest_path = Path(dest_dir)
        
        if not source_path.exists():
            print(f"⚠️  Source not found: {source_path}")
            continue
            
        dest_path.mkdir(parents=True, exist_ok=True)
        
        # Copy CSV files
        csv_files = list(source_path.glob("*.csv"))
        print(f"📁 Copying {len(csv_files)} files from {source_path} to {dest_path}")
        
        for csv_file in csv_files:
            dest_file = dest_path / csv_file.name
            if not dest_file.exists():
                shutil.copy2(csv_file, dest_file)
                print(f"  ✓ Copied: {csv_file.name}")
            else:
                print(f"  ⊘ Skipped (exists): {csv_file.name}")
    
    print("\n✅ Data preparation complete!")

if __name__ == "__main__":
    prepare_data()
