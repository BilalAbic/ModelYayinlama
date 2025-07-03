#!/usr/bin/env python3
"""
Alternative download script using gdown for Google Drive files.
Install: pip install gdown
"""

import os
import zipfile
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_with_gdown(file_id, destination):
    """Download using gdown library (install with: pip install gdown)."""
    try:
        import gdown
    except ImportError:
        raise ImportError("gdown library not found. Install with: pip install gdown")
    
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination, quiet=False)
    logger.info(f"Downloaded with gdown: {destination}")

def extract_zip(zip_path, extract_to):
    """Extract zip file to destination."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    logger.info(f"Extracted: {zip_path} -> {extract_to}")
    
    # Remove zip file after extraction
    os.remove(zip_path)
    logger.info(f"Cleaned up: {zip_path}")

def main():
    """Download using gdown method."""
    current_dir = Path(__file__).parent
    
    downloads = {
        "fine_tuned_model.zip": {
            "file_id": "1op6uXSfuXKEleLhZNQGcPWmsfbV9ocQL",
            "extract_to": current_dir,
            "folder_name": "fine_tuned_FitTurkAI_QLoRA"
        },
        "vector_store.zip": {
            "file_id": "16m5c9tOUVw_vwkUzxxt7HYL6mdKS2taV",
            "extract_to": current_dir,
            "folder_name": "fitness_rag_store_merged"
        }
    }
    
    # First try to install gdown
    try:
        import gdown
    except ImportError:
        print("📦 Installing gdown library...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown
    
    for filename, config in downloads.items():
        folder_path = current_dir / config["folder_name"]
        
        # Skip if folder already exists
        if folder_path.exists():
            logger.info(f"✅ {config['folder_name']} already exists, skipping download")
            continue
        
        logger.info(f"📥 Downloading {filename} with gdown...")
        zip_path = current_dir / filename
        
        try:
            download_with_gdown(config["file_id"], str(zip_path))
            extract_zip(str(zip_path), str(config["extract_to"]))
            logger.info(f"✅ {config['folder_name']} ready!")
        except Exception as e:
            logger.error(f"❌ Failed to download {filename}: {e}")
            # Clean up failed download
            if zip_path.exists():
                os.remove(str(zip_path))

if __name__ == "__main__":
    main() 