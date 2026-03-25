from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Optional, NewType, TYPE_CHECKING
from datetime import datetime
from numpy import typing as npt

if TYPE_CHECKING:
    from .object import DetectedObject

FrameId = NewType("FrameId", int)


@dataclass(kw_only=True, slots=True)
class Frame:
    """Base Frame Object, Contain metadata of a single frame or image."""

    id: FrameId
    width: int
    height: int
    image_name: str
    video_name: Optional[str] = None
    frame_idx: Optional[int] = None
    Timestamp: Optional[datetime] = None


@dataclass(slots=True)
class DetectedFrame(Frame):
    """Detected Frame Object, Contain detection results of a single frame or image."""

    detection_model: Optional[str] = None
    detected_objects: list[DetectedObject] = field(default_factory=list)

    def add_object(self, obj: DetectedObject) -> None:
        """Add a detected object to the frame."""
        self.detected_objects.append(obj)

    def get_obj_ids(self) -> npt.NDArray[np.int64]:
        """Get list of detected object IDs in the frame."""
        return np.array([obj.id for obj in self.detected_objects], dtype=np.int64)

    def get_obj_bboxes(self) -> npt.NDArray[np.float64]:
        """Get list of bounding boxes of detected objects in the frame."""
        return np.array(
            [obj.bounding_box.coord for obj in self.detected_objects], dtype=np.float64
        )
