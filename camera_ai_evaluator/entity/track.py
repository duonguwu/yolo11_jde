from typing import Any, Optional, NewType
from dataclasses import dataclass, field
import numpy as np

from .bounding_box import BoundingBox
from .object import DetectedObject, ClassId
from .frame import FrameId, DetectedFrame
from .sequence_info import SequenceInfo
from camera_ai_evaluator.entity import sequence_info

TrackId = NewType("TrackId", int)


@dataclass(slots=True)
class SingleTrack:
    """
    Information of one tracked object across multiple frames
    """

    track_id: TrackId
    frame_indices: list[FrameId] = field(default_factory=list)
    bboxes: list[BoundingBox] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    labels: list[Optional[ClassId]] = field(default_factory=list)
    features: list[Optional[np.ndarray]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_obj(self, frame_id: FrameId, obj: DetectedObject) -> None:
        """Add a detection to the track."""
        self.frame_indices.append(frame_id)
        self.bboxes.append(obj.bounding_box)
        self.confidences.append(obj.confidence)
        self.labels.append(obj.label)
        self.features.append(obj.feature)

    def latest_bbox(self) -> Optional[BoundingBox]:
        """Return the most recent bounding box."""
        return self.bboxes[-1] if self.bboxes else None


@dataclass(slots=True)
class Tracking:
    """Overall tracking result for a sequence or video."""

    sequence_info: SequenceInfo
    frames: dict[FrameId, DetectedFrame] = field(default_factory=dict)
    tracks: dict[TrackId, SingleTrack] = field(default_factory=dict)

    consider_masks: Optional[np.ndarray] = None
    """
    Consider masks use to apply other function to just only object inside its.
    """

    def add_detected_frame(self, frame: DetectedFrame) -> None:
        """Add a detected frame to the tracking result."""
        self.frames[frame.id] = frame

    def add_track(self, track: SingleTrack) -> None:
        """Add a completed track."""
        self.tracks[track.track_id] = track

    def get_total_frames(self) -> int:
        """Get total number of frames in the tracking result."""
        return len(self.frames)

    def get_total_tracks(self) -> int:
        """Get total number of tracks in the tracking result."""
        return len(self.tracks)

    def get_frame_obj(self, frame_id: FrameId) -> Optional[DetectedFrame]:
        """Get detected frame by its ID."""
        return self.frames.get(frame_id, None)
