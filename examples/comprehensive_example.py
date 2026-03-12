#!/usr/bin/env python3
"""
Comprehensive example script for YOLO-JDE Tracker.
Demonstrates all major features and usage patterns.
"""

import sys
import os
import json
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from yolo_jde import JDETracker, YOLOJDE


def example_1_basic_tracking():
    """Example 1: Basic video tracking with JDETracker."""
    print("🚀 Example 1: Basic Video Tracking")
    print("-" * 50)
    
    # Paths
    model_path = project_root / "yolo_jde" / "YOLO11s_JDE-CHMOT17.pt"
    video_path = "demo_video.mp4"
    output_path = "tracked_output.mp4"
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    # Create demo video if not exists
    if not os.path.exists(video_path):
        print("📹 Creating demo video...")
        create_demo_video(video_path)
    
    try:
        # Initialize tracker
        print("🤖 Initializing JDE Tracker...")
        tracker = JDETracker(
            model_path=str(model_path),
            tracker_config="smiletrack.yaml",
            mode="track",
            device="cuda:0" if torch.cuda.is_available() else "cpu"
        )
        
        # Run tracking
        print("🎯 Running tracking...")
        results = tracker.track_video(
            source=video_path,
            output_path=output_path,
            save_json=True,
            json_path="tracking_results.json",
            show_progress=True
        )
        
        # Print results
        print(f"✅ Tracking completed!")
        print(f"📹 Output video: {output_path}")
        print(f"📊 Processed {len(results)} frames")
        
        if results:
            total_detections = sum(len(frame['objects']) for frame in results)
            unique_ids = set()
            for frame in results:
                for obj in frame['objects']:
                    if obj['id'] > 0:
                        unique_ids.add(obj['id'])
            
            print(f"📈 Total detections: {total_detections}")
            print(f"🆔 Unique tracks: {len(unique_ids)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def example_2_yolo_jde_class():
    """Example 2: Using YOLOJDE class directly."""
    print("\n🚀 Example 2: YOLO-JDE Class Usage")
    print("-" * 50)
    
    model_path = project_root / "yolo_jde" / "YOLO11s_JDE-CHMOT17.pt"
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    try:
        # Initialize model
        print("🤖 Loading YOLO-JDE model...")
        model = YOLOJDE(str(model_path), task="jde")
        
        # Method 1: Track video
        print("🎯 Method 1: Track video...")
        track_results = model.track(
            source="demo_video.mp4",
            tracker="smiletrack.yaml",
            save=True,
            project="runs/track",
            name="yolojde_demo",
            exist_ok=True
        )
        
        print(f"✅ Tracking results: {len(list(track_results))} frames")
        
        # Method 2: Predict single image
        image_path = project_root / "tests" / "image.png"
        if image_path.exists():
            print("🔍 Method 2: Predict single image...")
            pred_results = model.predict(str(image_path), save=True)
            print(f"✅ Prediction completed: {len(pred_results)} results")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def example_3_detection_coco():
    """Example 3: Detection in COCO format."""
    print("\n🚀 Example 3: COCO Detection Format")
    print("-" * 50)
    
    model_path = project_root / "yolo_jde" / "YOLO11s_JDE-CHMOT17.pt"
    image_path = project_root / "tests" / "image.png"
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    if not image_path.exists():
        print(f"❌ Test image not found: {image_path}")
        return False
    
    try:
        # Initialize in detect mode
        print("🤖 Initializing detector...")
        tracker = JDETracker(
            model_path=str(model_path),
            mode="detect"
        )
        
        # Run detection
        print("🔍 Running COCO detection...")
        coco_results = tracker.detect_coco_format(
            source=str(image_path),
            image_ids=[1]
        )
        
        # Save results
        output_file = "coco_detection_results.json"
        with open(output_file, "w") as f:
            json.dump(coco_results, f, indent=2)
        
        print(f"✅ Detection completed!")
        print(f"📊 Found {len(coco_results)} objects")
        print(f"💾 Results saved to: {output_file}")
        
        # Show first few results
        for i, det in enumerate(coco_results[:3]):
            print(f"📦 Detection {i+1}:")
            print(f"   Category: {det['category_id']}")
            print(f"   BBox: {det['bbox']}")
            print(f"   Score: {det['score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def example_4_real_time_tracking():
    """Example 4: Real-time tracking from webcam."""
    print("\n🚀 Example 4: Real-time Webcam Tracking")
    print("-" * 50)
    
    model_path = project_root / "yolo_jde" / "YOLO11s_JDE-CHMOT17.pt"
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    try:
        # Initialize tracker
        print("🤖 Initializing real-time tracker...")
        tracker = JDETracker(
            model_path=str(model_path),
            tracker_config="smiletrack.yaml",
            mode="track"
        )
        
        # Try to open webcam
        print("📷 Opening webcam...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("⚠️ Cannot open webcam, using demo video instead...")
            cap = cv2.VideoCapture("demo_video.mp4")
            if not cap.isOpened():
                print("❌ Cannot open any video source")
                return False
        
        print("🎯 Starting real-time tracking...")
        print("Press 'q' to quit, 's' to save frame")
        
        frame_count = 0
        saved_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Track frame
            result = tracker.track_frame(frame)
            
            # Visualize
            annotated_frame = tracker.visualizer.draw_tracks(
                frame, 
                result['boxes'], 
                result.get('track_ids', [-1] * len(result['boxes'])), 
                result.get('scores')
            )
            
            # Add info text
            info_text = f"Frame: {frame_count}, Objects: {len(result['boxes'])}"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow("YOLO-JDE Real-time Tracking", annotated_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_path = f"saved_frame_{saved_frames:04d}.jpg"
                cv2.imwrite(save_path, annotated_frame)
                print(f"💾 Saved frame: {save_path}")
                saved_frames += 1
            
            frame_count += 1
            
            # Limit demo to 100 frames
            if frame_count >= 100:
                print("📊 Demo completed (100 frames processed)")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"✅ Real-time tracking completed!")
        print(f"📊 Processed {frame_count} frames")
        print(f"💾 Saved {saved_frames} frames")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def example_5_custom_config():
    """Example 5: Using custom tracker configuration."""
    print("\n🚀 Example 5: Custom Tracker Configuration")
    print("-" * 50)
    
    # Create custom config
    custom_config = {
        "tracker_type": "smiletrack",
        "track_high_thresh": 0.7,
        "track_low_thresh": 0.2,
        "new_track_thresh": 0.8,
        "track_buffer": 20,
        "match_thresh": 0.9,
        "proximity_thresh": 0.4,
        "appearance_thresh": 0.3,
        "with_reid": True,
        "lambda_": 0.95
    }
    
    # Save custom config
    import yaml
    config_path = "custom_tracker.yaml"
    with open(config_path, "w") as f:
        yaml.dump(custom_config, f, default_flow_style=False)
    
    print(f"📝 Created custom config: {config_path}")
    
    try:
        model_path = project_root / "yolo_jde" / "YOLO11s_JDE-CHMOT17.pt"
        
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            return False
        
        # Initialize with custom config
        print("🤖 Initializing with custom config...")
        tracker = JDETracker(
            model_path=str(model_path),
            tracker_config=config_path,
            mode="track"
        )
        
        # Test with single frame
        if os.path.exists("demo_video.mp4"):
            cap = cv2.VideoCapture("demo_video.mp4")
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                print("🎯 Testing custom config...")
                result = tracker.track_frame(frame)
                print(f"✅ Custom config test completed!")
                print(f"📊 Found {len(result['boxes'])} objects")
        
        # Clean up
        os.remove(config_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_demo_video(output_path, duration=10, fps=30):
    """Create a demo video with moving objects."""
    print(f"📹 Creating demo video: {output_path}")
    
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_idx in range(duration * fps):
        # Create background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # Dark gray background
        
        # Moving object 1 (person-like)
        x1 = int(100 + 200 * np.sin(frame_idx * 0.05))
        y1 = int(200 + 100 * np.cos(frame_idx * 0.03))
        cv2.rectangle(frame, (x1, y1), (x1+60, y1+120), (0, 255, 0), -1)
        cv2.circle(frame, (x1+30, y1-20), 20, (0, 255, 0), -1)  # Head
        
        # Moving object 2 (car-like)
        x2 = int(400 + 150 * np.cos(frame_idx * 0.04))
        y2 = int(300 + 80 * np.sin(frame_idx * 0.06))
        cv2.rectangle(frame, (x2, y2), (x2+100, y2+50), (255, 0, 0), -1)
        cv2.circle(frame, (x2+20, y2+50), 15, (0, 0, 0), -1)  # Wheel
        cv2.circle(frame, (x2+80, y2+50), 15, (0, 0, 0), -1)  # Wheel
        
        # Moving object 3 (smaller object)
        x3 = int(50 + 300 * (frame_idx / (duration * fps)))
        y3 = int(100 + 50 * np.sin(frame_idx * 0.1))
        cv2.rectangle(frame, (x3, y3), (x3+40, y3+40), (0, 0, 255), -1)
        
        # Add frame info
        cv2.putText(frame, f"Demo Frame {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "YOLO-JDE Tracking Demo", (10, height-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        out.write(frame)
    
    out.release()
    print(f"✅ Demo video created: {output_path} ({duration}s, {fps}fps)")


def main():
    """Run all examples."""
    print("🚀 YOLO-JDE Comprehensive Examples")
    print("=" * 70)
    
    # Check torch availability
    try:
        import torch
        device_info = f"CUDA: {torch.cuda.is_available()}"
        if torch.cuda.is_available():
            device_info += f" ({torch.cuda.get_device_name(0)})"
        print(f"🖥️  Device: {device_info}")
    except ImportError:
        print("⚠️  PyTorch not available")
    
    # Examples to run
    examples = [
        ("Basic Tracking", example_1_basic_tracking),
        ("YOLO-JDE Class", example_2_yolo_jde_class),
        ("COCO Detection", example_3_detection_coco),
        ("Real-time Tracking", example_4_real_time_tracking),
        ("Custom Config", example_5_custom_config)
    ]
    
    results = []
    
    for name, example_func in examples:
        try:
            success = example_func()
            results.append((name, success))
        except KeyboardInterrupt:
            print(f"\n⚠️ {name} interrupted by user")
            results.append((name, False))
            break
        except Exception as e:
            print(f"❌ {name} failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Examples Summary:")
    print("-" * 70)
    
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {name:<20} : {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\n🎯 Results: {passed}/{total} examples passed")
    
    if passed == total:
        print("🎉 All examples completed successfully!")
    else:
        print("⚠️ Some examples failed. Check the logs above.")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Examples interrupted by user")
        sys.exit(1)
