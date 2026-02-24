#!/usr/bin/env python
"""
Download Formula 1 Championships Dataset from Kaggle

This script downloads the Formula 1 Championships (1950-2025) dataset
from Kaggle using the kagglehub library.

Requirements:
    - kagglehub library installed
    - Kaggle API credentials configured
"""

import os
import kagglehub
from pathlib import Path

def download_f1_dataset():
    """
    Download the Formula 1 Championships dataset.
    
    Returns:
        str: Path to the downloaded dataset directory
    """
    print("Downloading Formula 1 Championships (1950-2025) dataset...")
    
    try:
        # Download latest version
        path = kagglehub.dataset_download("rockyt07/formula-1-championships-1950-2025")
        
        print(f"\n✓ Dataset downloaded successfully!")
        print(f"Path to dataset files: {path}")
        
        # List downloaded files
        dataset_path = Path(path)
        print(f"\nDataset contents:")
        for file in sorted(dataset_path.glob('*')):
            if file.is_file():
                size = file.stat().st_size / (1024 * 1024)  # Convert to MB
                print(f"  - {file.name} ({size:.2f} MB)")
        
        return path
        
    except Exception as e:
        print(f"\n✗ Error downloading dataset: {str(e)}")
        print("Make sure you have:")
        print("  1. Installed kagglehub: pip install kagglehub")
        print("  2. Configured Kaggle API credentials")
        raise

if __name__ == "__main__":
    dataset_path = download_f1_dataset()
    print(f"\nReady for analysis! Dataset location: {dataset_path}")
