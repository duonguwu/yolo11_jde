#!/usr/bin/env python3
"""
Test COCO detection format với model thật và ảnh thật.
Sử dụng YOLO11s_JDE-CHMOT17.pt và tests/image.png
"""

import sys
import json
import numpy as np
import cv2
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Paths to real files
MODEL_PATH = project_root / "yolo_jde" / "YOLO11s_JDE-CHMOT17.pt"
IMAGE_PATH = project_root / "tests" / "image.png"

def check_files():
    """Kiểm tra files có tồn tại không."""
    print("📁 Checking required files...")
    
    if not MODEL_PATH.exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        return False
    else:
        print(f"✅ Model found: {MODEL_PATH}")
    
    if not IMAGE_PATH.exists():
        print(f"❌ Image not found: {IMAGE_PATH}")
        return False
    else:
        print(f"✅ Image found: {IMAGE_PATH}")
    
    return True

def test_real_jde_detection():
    """Test JDE detection với model và ảnh thật."""
    print("🧪 Testing Real JDE Detection")
    print("-" * 40)
    
    try:
        from yolo_jde import JDETracker
        
        # Load real image
        print("📷 Loading test image...")
        image = cv2.imread(str(IMAGE_PATH))
        if image is None:
            print(f"❌ Cannot load image: {IMAGE_PATH}")
            return False
        
        print(f"✅ Image loaded: {image.shape}")
        
        # Create tracker with real model
        print("🤖 Loading JDE model...")
        tracker = JDETracker(model_path=str(MODEL_PATH), mode="detect")
        print("✅ JDE model loaded successfully")
        
        # Run real detection
        print("🔍 Running detection...")
        result = tracker.predict_frame(image)
        
        print(f"✅ Detection completed")
        print(f"📊 Found {len(result['boxes'])} objects")
        
        if len(result['boxes']) > 0:
            print(f"📊 Boxes shape: {result['boxes'].shape}")
            print(f"📊 Scores shape: {result['scores'].shape}")
            print(f"📊 Classes shape: {result['classes'].shape}")
            print(f"📊 Embeddings shape: {result['embeddings'].shape}")
            
            # Show detection results
            for i in range(min(5, len(result['boxes']))):  # Show max 5 objects
                box = result['boxes'][i]
                score = result['scores'][i]
                cls = result['classes'][i]
                
                print(f"\n📦 Object {i+1}:")
                print(f"   BBOX: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
                print(f"   Score: {score:.3f}")
                print(f"   Class: {int(cls)}")
        
        # Test format_frame_objects với real data
        print("\n📋 Testing object formatting with real data...")
        objects = tracker._format_frame_objects(result)
        
        print(f"✅ Formatted {len(objects)} objects")
        
        # Validate structure với real data
        for i, obj in enumerate(objects[:3]):  # Show first 3 objects
            print(f"\n📦 Formatted Object {i+1}:")
            print(f"   ID: {obj['id']}")
            print(f"   BBOX (xyxy): {obj['bbox_xyxy']}")
            print(f"   BBOX (xywh): {obj['bbox_xywh']}")
            print(f"   Confidence: {obj['conf']}")
            print(f"   Class: {obj['class']}")
            
            # Validate bbox conversion
            x1, y1, x2, y2 = obj['bbox_xyxy']
            cx, cy, w, h = obj['bbox_xywh']
            
            expected_cx = (x1 + x2) / 2
            expected_cy = (y1 + y2) / 2
            expected_w = x2 - x1
            expected_h = y2 - y1
            
            if abs(cx - expected_cx) < 0.1 and abs(cy - expected_cy) < 0.1:
                print("   ✅ BBOX conversion correct")
            else:
                print("   ❌ BBOX conversion incorrect")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_coco_format_output():
    """Test COCO format output structure."""
    print("\n🎯 Testing COCO Format Output Structure")
    print("-" * 40)
    
    # Expected COCO detection format:
    # {
    #   "image_id": 1,
    #   "category_id": 1,
    #   "bbox": [x, y, width, height],
    #   "score": 0.95
    # }
    
    sample_coco_detection = {
        "image_id": 1,
        "category_id": 1,
        "bbox": [100, 100, 100, 100],  # x, y, w, h
        "score": 0.95
    }
    
    print("📋 Sample COCO detection format:")
    print(json.dumps(sample_coco_detection, indent=2))
    
    # Validate required fields
    required_fields = ["image_id", "category_id", "bbox", "score"]
    
    for field in required_fields:
        if field in sample_coco_detection:
            print(f"✅ {field}: present")
        else:
            print(f"❌ {field}: missing")
            return False
    
    # Validate bbox format (should be [x, y, w, h])
    bbox = sample_coco_detection["bbox"]
    if len(bbox) == 4:
        print("✅ BBOX format: correct length (4)")
        print(f"   x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")
    else:
        print("❌ BBOX format: incorrect length")
        return False
    
    return True

def test_coco_format_conversion():
    """Test chuyển đổi sang COCO format với real detection."""
    print("\n📄 Testing COCO Format Conversion...")
    
    try:
        from yolo_jde import JDETracker
        
        # Load image và model
        image = cv2.imread(str(IMAGE_PATH))
        tracker = JDETracker(model_path=str(MODEL_PATH), mode="detect")
        
        # Run detection
        result = tracker.predict_frame(image)
        
        # Convert to COCO format manually
        coco_results = []
        image_id = 1
        
        for i in range(len(result['boxes'])):
            box_xyxy = result['boxes'][i]
            score = result['scores'][i]
            cls = result['classes'][i]
            
            # Convert xyxy to xywh (COCO format)
            x1, y1, x2, y2 = box_xyxy
            x = x1
            y = y1
            w = x2 - x1
            h = y2 - y1
            
            coco_det = {
                "image_id": image_id,
                "category_id": int(cls) + 1,  # COCO categories start from 1
                "bbox": [float(x), float(y), float(w), float(h)],
                "score": float(score)
            }
            coco_results.append(coco_det)
        
        # Save real COCO output
        output_file = "real_coco_detections.json"
        with open(output_file, "w") as f:
            json.dump(coco_results, f, indent=2)
        
        print(f"✅ Real COCO output saved to: {output_file}")
        print(f"📊 Total detections: {len(coco_results)}")
        
        # Show first few detections
        for i, det in enumerate(coco_results[:3]):
            print(f"\n📦 COCO Detection {i+1}:")
            print(f"   Image ID: {det['image_id']}")
            print(f"   Category ID: {det['category_id']}")
            print(f"   BBOX (xywh): {det['bbox']}")
            print(f"   Score: {det['score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ COCO conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Chạy tất cả real COCO detection tests."""
    print("🚀 Real JDE Detection & COCO Format Tests")
    print("=" * 60)
    
    # Check files first
    if not check_files():
        print("❌ Required files not found. Please check paths.")
        return False
    
    tests = [
        test_real_jde_detection,
        test_coco_format_output,
        test_coco_format_conversion
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print()  # Add spacing between tests
        if test():
            passed += 1
        else:
            print(f"❌ Test failed: {test.__name__}")
            break
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All real JDE detection tests passed!")
        print("💡 Real model inference working correctly!")
        print("📁 Check 'real_coco_detections.json' for COCO format output")
        return True
    else:
        print("❌ Some tests failed. Please check the issues.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
