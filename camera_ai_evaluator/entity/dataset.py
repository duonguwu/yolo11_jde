from dataclasses import dataclass, field

from .frame import DetectedFrame, FrameId, Frame

@dataclass(slots=True)
class Dataset:
    """Generic Dataset Object"""
    frames: dict[FrameId, Frame] = field(default_factory=dict)
    
    def add_frame(self, frame: Frame) -> None:
        """Add a frame to the dataset."""
        self.frames[frame.id] = frame

@dataclass(slots=True)
class DetectedDataset:
    """Detected Dataset Object, Contain detection results of multiple frames or images."""

    detected_frames: dict[FrameId, DetectedFrame] = field(default_factory=dict)

    def add_frame(self, frame: DetectedFrame) -> None:
        """Add a detected frame to the dataset."""
        self.detected_frames[frame.id] = frame

    def re_id_frames_by_image_name(self) -> None:
        """
        Re-id frames based on their image name.
        This method will sort frames by image name and reassign frame IDs accordingly.
        It also updates the frame_id of all detected objects within each frame
        to maintain data consistency.
        """

        # Sort frames by image_name (empty string if None)
        sorted_frames = sorted(
            self.detected_frames.values(), key=lambda f: f.image_name or ""
        )

        # Create a new dict with updated FrameIds
        new_detected_frames: dict[FrameId, DetectedFrame] = {}
        for i, frame in enumerate(sorted_frames):
            new_id = FrameId(i + 1)  # Start IDs from 1
            frame.id = new_id

            # Update frame_id for all detected objects in the frame
            for obj in frame.detected_objects:
                obj.frame_id = new_id

            new_detected_frames[frame.id] = frame

        # Replace old dict with reindexed one
        self.detected_frames = new_detected_frames

    def filter_by_score(self, score_threshold: float) -> "DetectedDataset":
        """
        Filter predictions by confidence score threshold.
        
        Args:
            score_threshold: Minimum confidence score to keep (predictions with score >= threshold are kept)
            
        Returns:
            New DetectedDataset with filtered predictions
        """
        filtered_dataset = DetectedDataset()
        
        for frame_id, frame in self.detected_frames.items():
            # Filter objects by confidence score
            filtered_objects = [
                obj for obj in frame.detected_objects 
                if obj.confidence >= score_threshold
            ]
            
            # Create new frame with filtered objects (preserve all frame attributes)
            filtered_frame = DetectedFrame(
                id=frame.id,
                width=frame.width,
                height=frame.height,
                image_name=frame.image_name,
                video_name=frame.video_name,
                frame_idx=frame.frame_idx,
                Timestamp=frame.Timestamp,
                detection_model=frame.detection_model,
                detected_objects=filtered_objects,
            )
            filtered_dataset.add_frame(filtered_frame)
        
        return filtered_dataset
