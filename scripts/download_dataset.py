"""
Download weapon detection dataset from Roboflow.

This script downloads the dataset in YOLOv8 format and saves it to the
project directory. It uses environment variables for API credentials.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
load_dotenv()

def download_dataset():
    """Download dataset from Roboflow."""
    
    # Get credentials from environment
    api_key = os.getenv('ROBOFLOW_API_KEY')
    workspace = os.getenv('ROBOFLOW_WORKSPACE')
    project = os.getenv('ROBOFLOW_PROJECT')
    version = os.getenv('ROBOFLOW_VERSION', '1')
    
    # Validate credentials
    if not api_key:
        print("Error: ROBOFLOW_API_KEY not found in .env file")
        return False
    
    print("Downloading dataset from Roboflow...")
    print(f"Project: {workspace}/{project} (v{version})")
    
    try:
        from roboflow import Roboflow
        
        rf = Roboflow(api_key=api_key)
        proj = rf.workspace(workspace).project(project)
        ver = proj.version(int(version))
        dataset = ver.download("yolov8")
        
        print(f"\nDataset downloaded to: {dataset.location}")
        
        # Display class information
        dataset_path = Path(dataset.location)
        data_yaml = dataset_path / "data.yaml"
        if data_yaml.exists():
            import yaml
            with open(data_yaml, 'r') as f:
                data_config = yaml.safe_load(f)
            
            if 'names' in data_config:
                names = data_config['names']
                if isinstance(names, list):
                    print(f"Classes ({len(names)}): {', '.join(names)}")
                elif isinstance(names, dict):
                    print(f"Classes ({len(names)}): {', '.join(names.values())}")
        
        return True
        
    except ImportError:
        print("Error: roboflow package not installed")
        print("Install with: pip install roboflow")
        return False
    
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False


if __name__ == "__main__":
    success = download_dataset()
    sys.exit(0 if success else 1)
