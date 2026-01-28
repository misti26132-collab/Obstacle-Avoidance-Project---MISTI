#!/usr/bin/env python3
"""
Validation script to check if all required packages are installed and working.
"""

import sys

def check(name, test_func):
    """Check if a package/feature works"""
    try:
        test_func()
        print(f"✅ {name}")
        return True
    except Exception as e:
        print(f"❌ {name}: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("  JETSON UPDATE FIX - VALIDATION CHECKLIST")
    print("="*60 + "\n")
    
    results = []
    
    # Check PyTorch
    results.append(check(
        "PyTorch",
        lambda: __import__('torch')
    ))
    
    # Check CUDA
    if results[-1]:
        import torch
        results.append(check(
            f"CUDA Available (Device: Orin)" if torch.cuda.is_available() else "CUDA",
            lambda: torch.cuda.is_available() or (_ for _ in ()).throw(
                RuntimeError("CUDA not available")
            )
        ))
        
        # Check CUDA version
        if torch.cuda.is_available():
            results.append(check(
                f"CUDA Device: {torch.cuda.get_device_name(0)}",
                lambda: None
            ))
    
    # Check TorchVision
    results.append(check(
        "TorchVision",
        lambda: __import__('torchvision')
    ))
    
    # Check NumPy
    results.append(check(
        "NumPy",
        lambda: __import__('numpy')
    ))
    
    # Check OpenCV
    results.append(check(
        "OpenCV",
        lambda: __import__('cv2')
    ))
    
    # Check ultralytics (YOLO)
    results.append(check(
        "ultralytics (YOLO)",
        lambda: __import__('ultralytics')
    ))
    
    # Check pyttsx3 (Text-to-Speech)
    results.append(check(
        "pyttsx3 (Speech)",
        lambda: __import__('pyttsx3')
    ))
    
    # Check project modules
    results.append(check(
        "camera_utils",
        lambda: __import__('camera_utils')
    ))
    
    results.append(check(
        "depth module",
        lambda: __import__('depth')
    ))
    
    results.append(check(
        "config module",
        lambda: __import__('config')
    ))
    
    # Summary
    print("\n" + "="*60)
    passed = sum(results)
    total = len(results)
    print(f"  RESULTS: {passed}/{total} checks passed")
    print("="*60 + "\n")
    
    if passed == total:
        print("✅ All systems ready! Run: python3 main.py\n")
        return 0
    else:
        print("⚠️  Some components missing. Run: python3 fix_pytorch.py\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
