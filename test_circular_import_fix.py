#!/usr/bin/env python3
"""
Test script to verify the circular import fixes work correctly.
This tests the core initialization without requiring a camera.
"""

import sys
import logging
import time

logging.basicConfig(level=logging.WARNING, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that imports work with circular import handling"""
    print("=" * 70)
    print("TEST 1: Core Imports (Circular Import Check)")
    print("=" * 70)
    
    try:
        # Pre-load torchvision as done in main.py
        print("[TEST] Pre-loading torchvision...")
        import torchvision
        import torchvision._meta_registrations
        print("✅ Torchvision pre-loaded successfully")
    except Exception as e:
        print(f"❌ Torchvision pre-load failed: {e}")
        return False
    
    try:
        print("[TEST] Importing project modules...")
        import config
        from depth import DepthEstimator
        from Speech_new import SpeechEngine
        print("✅ Project modules imported successfully")
    except Exception as e:
        print(f"❌ Project imports failed: {e}")
        return False
    
    return True

def test_depth_estimator():
    """Test depth estimator initialization with fallback"""
    print("\n" + "=" * 70)
    print("TEST 2: Depth Estimator Initialization (Fallback Check)")
    print("=" * 70)
    
    try:
        import config
        from depth import DepthEstimator
        
        print("[TEST] Initializing DepthEstimator...")
        depth = DepthEstimator()
        print("✅ DepthEstimator initialized successfully")
        
        # Test that it can process a dummy frame
        import numpy as np
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        print("[TEST] Processing dummy frame...")
        depth_map = depth.estimate(dummy_frame)
        print(f"✅ Depth estimation works, output shape: {depth_map.shape}")
        
        if depth_map is not None and depth_map.shape == (480, 640):
            print("✅ Depth output format is correct")
            return True
        else:
            print(f"❌ Depth output format is wrong: {depth_map.shape if depth_map is not None else 'None'}")
            return False
            
    except Exception as e:
        print(f"❌ Depth estimator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_yolo_fallback():
    """Test YOLO fallback mechanism"""
    print("\n" + "=" * 70)
    print("TEST 3: YOLO Fallback Mechanism (Circular Import Handling)")
    print("=" * 70)
    
    try:
        print("[TEST] Testing YOLO lazy import...")
        # This will test get_yolo() function
        from main import get_yolo
        
        print("[TEST] Getting YOLO model...")
        YOLO = get_yolo()
        
        print("[TEST] Instantiating YOLO with fallback...")
        yolo = YOLO('yolov8n.pt')
        
        print(f"✅ YOLO object created: {type(yolo).__name__}")
        
        if hasattr(yolo, 'predict'):
            # Test predict method
            import numpy as np
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            print("[TEST] Testing YOLO prediction...")
            results = yolo.predict(dummy_frame)
            print(f"✅ YOLO prediction works, returned {len(results)} result(s)")
            return True
        else:
            print("❌ YOLO object missing predict method")
            return False
            
    except Exception as e:
        print(f"❌ YOLO fallback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_detector_initialization():
    """Test detector with dummy models"""
    print("\n" + "=" * 70)
    print("TEST 4: Dual Model Detector Initialization (No Camera)")
    print("=" * 70)
    
    try:
        import config
        from depth import DepthEstimator
        from main import DualModelDetector
        
        print("[TEST] Initializing DepthEstimator...")
        depth = DepthEstimator()
        
        print("[TEST] Initializing DualModelDetector...")
        detector = DualModelDetector(depth)
        
        print("✅ Detector initialized successfully")
        
        # Test that detector can process frames
        import numpy as np
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        print("[TEST] Processing test frame with detector...")
        annotated_frame, direction, distance, obstacle_class, priority = detector.process(dummy_frame)
        
        print(f"✅ Detector.process() works")
        print(f"   - Output frame shape: {annotated_frame.shape}")
        print(f"   - Direction: {direction}, Distance: {distance}")
        print(f"   - Obstacle: {obstacle_class}, Priority: {priority}")
        
        return True
        
    except Exception as e:
        print(f"❌ Detector initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "CIRCULAR IMPORT FIX - VALIDATION TESTS" + " " * 15 + "║")
    print("╚" + "=" * 68 + "╝")
    
    results = {
        "Core Imports": test_imports(),
        "Depth Estimator": test_depth_estimator(),
        "YOLO Fallback": test_yolo_fallback(),
        "Detector Init": test_detector_initialization(),
    }
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<30} {status}")
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    print("=" * 70)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed! Circular import fixes are working correctly.")
        print("   The application can now initialize despite torch.hub/torchvision")
        print("   incompatibilities by using fallback mechanisms.")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please review the errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
