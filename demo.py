#!/usr/bin/env python3
"""
Demo script to test YOLO-JDE tracker module.
"""

import sys
from pathlib import Path

# Add module to path
sys.path.insert(0, str(Path(__file__).parent))

def test_import():
    """Test if module can be imported successfully."""
    try:
        from yolo_jde import JDETracker
        print("✓ Module imported successfully!")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_config():
    """Test configuration loading."""
    try:
        from yolo_jde.utils.config import ConfigManager
        
        config_manager = ConfigManager()
        
        # List available configs
        trackers = config_manager.list_configs('trackers')
        models = config_manager.list_configs('models')
        
        print(f"✓ Available trackers: {trackers}")
        print(f"✓ Available models: {models}")
        
        # Load a config
        if 'smiletrack' in trackers:
            config = config_manager.load_tracker_config('smiletrack')
            print(f"✓ SmileTrack config loaded: {config.get('tracker_type')}")
        
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def test_tracker_init():
    """Test tracker initialization without model."""
    try:
        from yolo_jde import JDETracker
        
        # Test detection-only mode (no model needed)
        tracker = JDETracker(mode="detect")
        print("✓ Tracker initialized in detection mode")
        
        return True
    except Exception as e:
        print(f"✗ Tracker init failed: {e}")
        return False

def test_components():
    """Test individual components."""
    try:
        # Test post-processor
        from yolo_jde.core.postprocess import PostProcessor
        postprocessor = PostProcessor()
        print("✓ PostProcessor created")
        
        # Test visualizer
        from yolo_jde.core.visualization import Visualizer
        visualizer = Visualizer()
        print("✓ Visualizer created")
        
        # Test Kalman filter
        from yolo_jde.trackers.utils.kalman_filter import KalmanFilterXYAH
        kf = KalmanFilterXYAH()
        print("✓ Kalman filter created")
        
        # Test matching utilities
        from yolo_jde.trackers.utils.matching import linear_assignment
        print("✓ Matching utilities available")
        
        return True
    except Exception as e:
        print(f"✗ Components test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("YOLO-JDE Tracker Module Demo")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_import),
        ("Configuration Test", test_config),
        ("Tracker Initialization", test_tracker_init),
        ("Components Test", test_components),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Module is ready to use.")
        print("\nNext steps:")
        print("1. Get a JDE model file (e.g., YOLO11s_JDE-CHMOT17.pt)")
        print("2. Run tracking: python examples/test_tracking.py --model model.pt --source video.mp4")
        print("3. Run detection: python examples/test_detection_coco.py --model model.pt --source images/")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
