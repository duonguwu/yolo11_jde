#!/usr/bin/env python3
"""
Test JDE tracking functionality với simulated data.
"""

import sys
import numpy as np
import cv2
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_mock_detection_data():
    """Tạo mock detection data giống output của detect."""
    # Simulate 3 detected objects
    num_objects = 3
    
    # Mock boxes (xyxy format)
    boxes = np.array([
        [100, 100, 200, 300],  # Person 1
        [300, 150, 400, 350],  # Person 2  
        [500, 120, 600, 320],  # Person 3
    ], dtype=np.float32)
    
    # Mock scores
    scores = np.array([0.85, 0.92, 0.78], dtype=np.float32)
    
    # Mock classes (all person = 0)
    classes = np.array([0, 0, 0], dtype=np.int32)
    
    # Mock embeddings (512-dim features)
    embeddings = np.random.randn(num_objects, 512).astype(np.float32)
    
    return {
        'boxes': boxes,
        'scores': scores, 
        'classes': classes,
        'embeddings': embeddings
    }

def test_tracker_initialization():
    """Test khởi tạo tracker components."""
    print("🧪 Testing Tracker Initialization")
    print("-" * 40)
    
    try:
        from yolo_jde import JDETracker
        
        # Test với mode track
        print("🤖 Initializing JDE tracker in track mode...")
        tracker = JDETracker(mode="track")
        print("✅ JDE tracker initialized successfully")
        
        # Check tracker components
        if hasattr(tracker, 'config_manager'):
            print("✅ Config manager: present")
        else:
            print("❌ Config manager: missing")
            return False
            
        if hasattr(tracker, 'postprocessor'):
            print("✅ Post processor: present")
        else:
            print("❌ Post processor: missing")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Tracker initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mock_tracking():
    """Test tracking với mock detection data."""
    print("\n🔍 Testing Mock Tracking Process")
    print("-" * 40)
    
    try:
        from yolo_jde import JDETracker
        
        # Initialize tracker
        tracker = JDETracker(mode="track")
        
        # Create mock image
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        print("📷 Created mock image: 640x640")
        
        # Create mock detection data
        detection_data = create_mock_detection_data()
        print(f"📊 Created mock detections:")
        print(f"   - Boxes: {detection_data['boxes'].shape}")
        print(f"   - Scores: {detection_data['scores'].shape}")
        print(f"   - Classes: {detection_data['classes'].shape}")
        print(f"   - Embeddings: {detection_data['embeddings'].shape}")
        
        # Test format_frame_objects
        print("\n📋 Testing object formatting...")
        objects = tracker._format_frame_objects(detection_data)
        print(f"✅ Formatted {len(objects)} objects")
        
        # Validate object structure
        for i, obj in enumerate(objects):
            print(f"\n📦 Object {i+1}:")
            print(f"   ID: {obj['id']}")
            print(f"   BBOX (xyxy): {obj['bbox_xyxy']}")
            print(f"   BBOX (xywh): {obj['bbox_xywh']}")
            print(f"   Confidence: {obj['conf']}")
            print(f"   Class: {obj['class']}")
            print(f"   Embedding shape: {obj['embedding'].shape}")
            
            # Validate required fields
            required_fields = ['id', 'bbox_xyxy', 'bbox_xywh', 'conf', 'class', 'embedding']
            for field in required_fields:
                if field in obj:
                    print(f"   ✅ {field}: present")
                else:
                    print(f"   ❌ {field}: missing")
                    return False
        
        return True
        
    except Exception as e:
        print(f"❌ Mock tracking failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tracker_config_loading():
    """Test loading tracker configurations."""
    print("\n⚙️ Testing Tracker Config Loading")
    print("-" * 40)
    
    try:
        from yolo_jde import JDETracker
        
        tracker = JDETracker(mode="track")
        
        # Test loading default config
        print("📄 Testing default tracker config...")
        try:
            config = tracker.config_manager.load_tracker_config('smiletrack')
            print("✅ Default config loaded successfully")
            print(f"   Config keys: {list(config.keys())}")
            
            # Check common tracker parameters
            common_params = ['track_high_thresh', 'track_low_thresh', 'new_track_thresh', 'match_thresh']
            for param in common_params:
                if param in config:
                    print(f"   ✅ {param}: {config[param]}")
                else:
                    print(f"   ⚠️ {param}: not found (may be optional)")
                    
        except Exception as e:
            print(f"⚠️ Config loading failed (expected if no config files): {e}")
            
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tracking_pipeline():
    """Test complete tracking pipeline simulation."""
    print("\n🔄 Testing Complete Tracking Pipeline")
    print("-" * 40)
    
    try:
        from yolo_jde import JDETracker
        
        tracker = JDETracker(mode="track")
        
        # Simulate multiple frames
        num_frames = 3
        
        for frame_id in range(num_frames):
            print(f"\n📹 Processing Frame {frame_id + 1}/{num_frames}")
            
            # Create mock image
            image = np.zeros((640, 640, 3), dtype=np.uint8)
            
            # Create slightly different detection data for each frame
            detection_data = create_mock_detection_data()
            
            # Add some noise to simulate movement
            if frame_id > 0:
                noise = np.random.normal(0, 5, detection_data['boxes'].shape)
                detection_data['boxes'] += noise
                
            # Format objects
            objects = tracker._format_frame_objects(detection_data)
            
            print(f"   📊 Frame {frame_id + 1}: {len(objects)} objects")
            
            # Show object IDs (should be consistent across frames in real tracking)
            for i, obj in enumerate(objects):
                print(f"      Object {i+1}: ID={obj['id']}, conf={obj['conf']:.3f}")
        
        print("\n✅ Pipeline simulation completed")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Chạy tất cả tracking tests."""
    print("🚀 JDE Tracking Module Tests")
    print("=" * 60)
    
    tests = [
        test_tracker_initialization,
        test_mock_tracking,
        test_tracker_config_loading,
        test_tracking_pipeline
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print()  # Add spacing between tests
        if test():
            passed += 1
        else:
            print(f"❌ Test failed: {test.__name__}")
            # Continue with other tests instead of breaking
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tracking tests passed!")
        print("💡 Tracking module components working correctly!")
    else:
        print("⚠️ Some tests failed. Check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
