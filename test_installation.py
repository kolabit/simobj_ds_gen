#!/usr/bin/env python3
"""
Test script to verify SAM2 YOLO annotation generator installation.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing module imports...")
    
    modules_to_test = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("sklearn", "scikit-learn"),
        ("tqdm", "tqdm"),
    ]
    
    failed_imports = []
    
    for module_name, display_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ {display_name} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {display_name}: {e}")
            failed_imports.append(module_name)
    
    # Test SAM2 specifically
    try:
        from segment_anything_2 import sam_model_registry
        print("✅ Segment Anything 2 imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Segment Anything 2: {e}")
        print("   This is expected if you haven't installed it yet.")
        failed_imports.append("segment_anything_2")
    
    return failed_imports

def test_local_modules():
    """Test if local project modules can be imported."""
    print("\nTesting local module imports...")
    
    local_modules = [
        ("utils", "utils.py"),
        ("config", "config.py"),
        ("yolo_annotator", "yolo_annotator.py"),
        ("object_detector", "object_detector.py"),
        ("sam2_processor", "sam2_processor.py"),
    ]
    
    failed_modules = []
    
    for module_name, file_name in local_modules:
        file_path = Path(file_name)
        if file_path.exists():
            try:
                __import__(module_name)
                print(f"✅ {module_name} imported successfully")
            except ImportError as e:
                print(f"❌ Failed to import {module_name}: {e}")
                failed_modules.append(module_name)
        else:
            print(f"❌ {file_name} not found")
            failed_modules.append(module_name)
    
    return failed_modules

def test_cuda_availability():
    """Test CUDA availability."""
    print("\nTesting CUDA availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA is available")
            print(f"   Devices: {device_count}")
            print(f"   Device 0: {device_name}")
            return True
        else:
            print("⚠️  CUDA is not available (will use CPU)")
            return False
    except ImportError:
        print("❌ PyTorch not available, cannot test CUDA")
        return False

def test_directory_structure():
    """Test if required files exist."""
    print("\nTesting project structure...")
    
    required_files = [
        "main.py",
        "requirements.txt",
        "README.md",
        "config.py",
    ]
    
    missing_files = []
    
    for file_name in required_files:
        if Path(file_name).exists():
            print(f"✅ {file_name} found")
        else:
            print(f"❌ {file_name} missing")
            missing_files.append(file_name)
    
    return missing_files

def test_python_version():
    """Test Python version compatibility."""
    print("\nTesting Python version...")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version is compatible (3.8+)")
        return True
    else:
        print("❌ Python version should be 3.8 or higher")
        return False

def main():
    """Run all tests."""
    print("SAM2 YOLO Annotation Generator - Installation Test")
    print("=" * 55)
    
    # Test Python version
    python_ok = test_python_version()
    
    # Test imports
    failed_imports = test_imports()
    
    # Test local modules
    failed_modules = test_local_modules()
    
    # Test CUDA
    cuda_available = test_cuda_availability()
    
    # Test project structure
    missing_files = test_directory_structure()
    
    # Summary
    print("\n" + "=" * 55)
    print("TEST SUMMARY")
    print("=" * 55)
    
    all_passed = True
    
    if not python_ok:
        print("❌ Python version issue")
        all_passed = False
    
    if failed_imports:
        print(f"❌ {len(failed_imports)} import failures")
        all_passed = False
    
    if failed_modules:
        print(f"❌ {len(failed_modules)} local module failures")
        all_passed = False
    
    if missing_files:
        print(f"❌ {len(missing_files)} missing files")
        all_passed = False
    
    if all_passed:
        print("✅ All tests passed!")
        print("\n🎉 Installation appears to be successful!")
        print("\nNext steps:")
        print("1. Add some images to a directory")
        print("2. Run: python main.py --input_dir images --output_dir annotations")
        print("3. Check the generated annotation files")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon solutions:")
        if failed_imports:
            print("- Install missing dependencies: pip install -r requirements.txt")
        if failed_modules:
            print("- Make sure all project files are in the same directory")
        if missing_files:
            print("- Download or recreate missing project files")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 