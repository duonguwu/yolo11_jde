import json
import numpy as np
from typing import Any, Dict, List
from typing_extensions import override

from camera_ai_evaluator.entity.object import (
    ClassId,
    AttributeName,
    ObjectId,
    DetectedObject,
)
from camera_ai_evaluator.entity.frame import (
    DetectedFrame,
    FrameId,
)
from camera_ai_evaluator.entity.dataset import DetectedDataset
from camera_ai_evaluator.entity.bounding_box import BoundingBox
from camera_ai_evaluator.annotation_io.base import DetectionAnnotationIO


class CocoDetectionAnnotationIO(DetectionAnnotationIO):
    """
    Handles loading and saving annotations in COCO (Common Objects in Context) format.
    """

    def __init__(self) -> None:
        super().__init__()

    @override
    def load_annotations(self, annotation_file_path: str) -> DetectedDataset:
        """
        Parses a COCO JSON annotation file and returns a DetectedDataset.

        Args:
            annotation_file_path: The path to the COCO JSON annotation file.

        Returns:
            A DetectedDataset object containing all the frames and their
            detected objects.
        """
        try:
            with open(annotation_file_path, "r") as f:
                coco_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(
                f"Error reading or parsing JSON file: {annotation_file_path}"
            ) from e

        # 1. Create a mapping from image_id to image info (width, height)
        images_info: Dict[int, Dict[str, Any]] = {
            img["id"]: img for img in coco_data.get("images", [])
        }

        # 2. Group annotations by image_id
        # COCO's annotation `id` is unique per annotation, not per object track.
        # We will use it as the `ObjectId`.
        annotations_by_image: Dict[int, List[Dict[str, Any]]] = {}
        for ann in coco_data.get("annotations", []):
            image_id = ann.get("image_id")
            if image_id is None:
                continue
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)

        # 3. Construct DetectedDataset
        detected_dataset = DetectedDataset()

        # Iterate through images to ensure frames without annotations are also included
        for image_id, image_info in images_info.items():
            frame_id = FrameId(image_id)
            width = image_info.get("width", 0)
            height = image_info.get("height", 0)
            image_name = image_info.get("file_name", "")

            detected_objects: List[DetectedObject] = []

            # Get annotations for the current image
            annotations_for_this_image = annotations_by_image.get(image_id, [])

            for ann in annotations_for_this_image:
                # COCO bbox format is [x_min, y_min, width, height]
                bbox_xywh = ann.get("bbox")
                if not bbox_xywh or len(bbox_xywh) != 4:
                    continue

                # The BoundingBox class should handle conversion from 'xywh'
                bbox = BoundingBox(
                    coord=np.array(bbox_xywh, dtype=np.float32),
                    coord_type="xywh",
                )

                # Extract other standard COCO fields
                object_id = ObjectId(ann["id"])
                class_id = ClassId(ann["category_id"])
                confidence = float(ann.get("score", 1.0))  # Use 1.0 for ground truth

                # Extract non-standard fields as attributes
                attributes: dict[AttributeName, Any] = {}
                standard_keys = {
                    "id",
                    "image_id",
                    "category_id",
                    "bbox",
                    "area",
                    "iscrowd",
                    "segmentation",
                    "score",
                }
                for key, value in ann.items():
                    if key == "attributes":
                        attributes = value

                detected_object = DetectedObject(
                    id=object_id,
                    frame_id=frame_id,
                    class_id=class_id,
                    confidence=confidence,
                    bounding_box=bbox,
                    attributes=attributes,
                )
                detected_objects.append(detected_object)

            # Create a DetectedFrame, even if it has no objects
            detected_frame = DetectedFrame(
                id=frame_id,
                width=width,
                height=height,
                detected_objects=detected_objects,
                image_name=image_name,
            )
            detected_dataset.add_frame(detected_frame)

        return detected_dataset

    @override
    def save_annotations(self, annotations: DetectedDataset, output_path: str) -> None:
        """
        Save annotations to a COCO JSON file.

        Args:
            annotations: A DetectedDataset object to be saved.
            output_path: The path to save the output JSON file.

        Raises:
            NotImplementedError: This method is not implemented yet.
        """
        raise NotImplementedError(
            "Saving annotations to COCO JSON format is not implemented yet."
        )


if __name__ == "__main__":
    import os

    # Now, run the loader
    annotation_io = CocoDetectionAnnotationIO()
    file_test_path = "data/test_annotation/detection/tran_duy_hung.json"

    try:
        dataset = annotation_io.load_annotations(file_test_path)

        # Sort frames by ID for consistent output
        sorted_frames = sorted(dataset.detected_frames.values(), key=lambda f: f.id)

        for detected_frame in sorted_frames:
            print(
                f"Frame ID: {detected_frame.id}, Size: {detected_frame.width}x{detected_frame.height}"
            )
            for obj in detected_frame.detected_objects:
                # The BoundingBox class should have a representation for printing
                # Assuming it has a __str__ or __repr__ method
                print(
                    f"  Object ID: {obj.id}, Class ID: {obj.class_id}, "
                    f"BBox (xywh): {obj.bounding_box.coord}, "
                    f"Confidence: {obj.confidence}, "
                    f"Attributes: {obj.attributes}"
                )
    except Exception as e:
        print(f"An error occurred: {e}")
