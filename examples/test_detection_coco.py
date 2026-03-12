#!/usr/bin/env python3
"""
Test script for detection-only mode with COCO format output.
"""

import argparse
import json
from pathlib import Path
import sys

# Add module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from yolo_jde import JDETracker


def main():
    parser = argparse.ArgumentParser(description="YOLO-JDE Detection Test (COCO Format)")
    parser.add_argument("--model", type=str, required=True, help="Path to JDE model weights")
    parser.add_argument("--source", type=str, required=True, 
                       help="Path to image file, directory, or list of images")
    parser.add_argument("--output", type=str, help="Path to output JSON file")
    parser.add_argument("--device", type=str, default="", help="Device to run on")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--image-ids", type=str, help="JSON file with image ID mapping")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return
    
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: Source not found: {args.source}")
        return
    
    # Set output path if not provided
    if not args.output:
        if source_path.is_file():
            args.output = str(source_path.parent / f"{source_path.stem}_detections.json")
        else:
            args.output = str(source_path / "detections.json")
    
    print(f"Source: {args.source}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device or 'auto'}")
    
    try:
        # Initialize detector (detection-only mode)
        detector = JDETracker(
            model_path=args.model,
            device=args.device,
            mode="detect",  # Detection-only mode
            model_config={
                'conf_thres': args.conf_thres,
                'iou_thres': args.iou_thres
            }
        )
        
        print("Detector initialized successfully!")
        
        # Load image IDs if provided
        image_ids = None
        if args.image_ids and Path(args.image_ids).exists():
            with open(args.image_ids, 'r') as f:
                image_id_mapping = json.load(f)
            # Convert to list if it's a mapping
            if isinstance(image_id_mapping, dict):
                image_ids = list(image_id_mapping.values())
            else:
                image_ids = image_id_mapping
        
        # Run detection
        coco_results = detector.detect_coco_format(
            source=args.source,
            image_ids=image_ids
        )
        
        print(f"Detection completed! Found {len(coco_results)} detections.")
        
        # Create COCO format output
        coco_output = {
            "images": [],
            "annotations": coco_results,
            "categories": [
                {"id": i, "name": f"class_{i}", "supercategory": "object"} 
                for i in range(80)  # COCO has 80 classes
            ]
        }
        
        # Add image info if processing directory
        if source_path.is_dir():
            image_files = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
            image_files.sort()
            
            for i, img_path in enumerate(image_files):
                img_id = image_ids[i] if image_ids and i < len(image_ids) else i
                coco_output["images"].append({
                    "id": img_id,
                    "file_name": img_path.name,
                    "width": 640,  # Default, should be actual image size
                    "height": 640
                })
        elif source_path.is_file():
            img_id = image_ids[0] if image_ids else 0
            coco_output["images"].append({
                "id": img_id,
                "file_name": source_path.name,
                "width": 640,  # Default, should be actual image size
                "height": 640
            })
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(coco_output, f, indent=2)
        
        print(f"Results saved to: {args.output}")
        
        # Print statistics
        if coco_results:
            scores = [det['score'] for det in coco_results]
            categories = set(det['category_id'] for det in coco_results)
            
            print(f"Detection statistics:")
            print(f"  Total detections: {len(coco_results)}")
            print(f"  Unique categories: {len(categories)}")
            print(f"  Score range: {min(scores):.3f} - {max(scores):.3f}")
            print(f"  Average score: {sum(scores)/len(scores):.3f}")
        
    except Exception as e:
        print(f"Error during detection: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
