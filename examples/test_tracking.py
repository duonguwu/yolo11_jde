#!/usr/bin/env python3
"""
Test script for video tracking using YOLO-JDE tracker.
Similar to track_save.py but using the standalone module.
"""

import argparse
from pathlib import Path
import sys

# Add module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from yolo_jde import JDETracker


def main():
    parser = argparse.ArgumentParser(description="YOLO-JDE Video Tracking Test")
    parser.add_argument("--model", type=str, required=True, help="Path to JDE model weights")
    parser.add_argument("--source", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, help="Path to output video (optional)")
    parser.add_argument("--tracker", type=str, default="smiletrack.yaml", 
                       help="Tracker configuration (smiletrack.yaml, bytetrack.yaml)")
    parser.add_argument("--device", type=str, default="", help="Device to run on")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--save-json", action="store_true", help="Save results as JSON")
    parser.add_argument("--json-path", type=str, help="JSON output path")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return
    
    if not Path(args.source).exists():
        print(f"Error: Video file not found: {args.source}")
        return
    
    # Set output path if not provided
    if not args.output:
        source_path = Path(args.source)
        args.output = str(source_path.parent / f"{source_path.stem}_tracked.mp4")
    
    if not args.json_path and args.save_json:
        output_path = Path(args.output)
        args.json_path = str(output_path.parent / f"{output_path.stem}_results.json")
    
    print(f"Input video: {args.source}")
    print(f"Output video: {args.output}")
    print(f"Model: {args.model}")
    print(f"Tracker: {args.tracker}")
    print(f"Device: {args.device or 'auto'}")
    
    try:
        # Initialize tracker
        tracker = JDETracker(
            model_path=args.model,
            tracker_config=args.tracker,
            device=args.device,
            mode="track",
            model_config={
                'conf_thres': args.conf_thres,
                'iou_thres': args.iou_thres
            }
        )
        
        print("Tracker initialized successfully!")
        
        # Run tracking
        results = tracker.track_video(
            source=args.source,
            output_path=args.output,
            save_json=args.save_json,
            json_path=args.json_path,
            show_progress=True
        )
        
        print(f"Tracking completed! Processed {len(results)} frames.")
        
        # Print statistics
        total_objects = sum(len(frame['objects']) for frame in results)
        unique_ids = set()
        for frame in results:
            for obj in frame['objects']:
                if obj['id'] >= 0:
                    unique_ids.add(obj['id'])
        
        print(f"Total detections: {total_objects}")
        print(f"Unique track IDs: {len(unique_ids)}")
        
    except Exception as e:
        print(f"Error during tracking: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
