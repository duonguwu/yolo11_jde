from dataclasses import dataclass, field
from typing import Literal, Optional, NewType, Any
import numpy as np

from camera_ai_evaluator.entity.bounding_box import BoundingBox
from camera_ai_evaluator.entity.frame import FrameId

AttributeName = NewType("AttributeName", str)

ObjectId = NewType("ObjectId", int)
ClassId = NewType("ClassId", int)


@dataclass(kw_only=True)
class Object:
    """Base class for an object detected or annotated in a frame."""

    id: ObjectId
    frame_id: FrameId
    class_id: ClassId
    attributes: dict[AttributeName, Any] = field(default_factory=dict)


@dataclass
class DetectedObject(Object):
    """Detection Object"""

    confidence: float
    bounding_box: BoundingBox
    label: Optional[ClassId] = None
    feature: Optional[np.ndarray] = None
