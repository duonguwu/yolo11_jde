import numpy as np
from numpy.typing import NDArray

from dataclasses import dataclass
from typing import Literal


def xywh_to_xyxy(boxes_xywh: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Convert COCO-format bounding boxes from [x_min, y_min, w, h]
    to [x1, y1, x2, y2].

    Args:
        boxes_xywh: Array of shape (N, 4) or (4,) with dtype float32.

    Returns:
        NDArray[np.float32]: Converted boxes with shape (N, 4) or (4,).
    """
    boxes = np.asarray(boxes_xywh, dtype=np.float32)
    single_box = False

    if boxes.ndim == 1:
        boxes = boxes[None, :]
        single_box = True

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    result = np.stack((x1, y1, x2, y2), axis=1)
    return result[0] if single_box else result


def xyxy_to_xywh(boxes_xyxy: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Convert bounding boxes from [x1, y1, x2, y2]
    to COCO format [x_min, y_min, w, h].

    Args:
        boxes_xyxy: Array of shape (N, 4) or (4,) with dtype float32.

    Returns:
        NDArray[np.float32]: Converted boxes in COCO format.
    """
    boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    single_box = False

    if boxes.ndim == 1:
        boxes = boxes[None, :]
        single_box = True

    x_min = boxes[:, 0]
    y_min = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    result = np.stack((x_min, y_min, w, h), axis=1)
    return result[0] if single_box else result


@dataclass
class BoundingBox:
    """Single bounding box."""

    coord: np.ndarray
    coord_type: Literal["xyxy", "xywh"] = "xyxy"

    def change_coord_type(self, new_type: Literal["xyxy", "xywh"]) -> None:
        if new_type != self.coord_type:
            self.coord_type = new_type
            if new_type == "xyxy":
                self.coord = xywh_to_xyxy(self.coord)
            else:
                self.coord = xyxy_to_xywh(self.coord)

    def area(self) -> float:
        """Calculate area of the bounding box."""
        if self.coord_type == "xyxy":
            x_min, y_min, x_max, y_max = self.coord
            width = max(0, x_max - x_min)
            height = max(0, y_max - y_min)
        elif self.coord_type == "xywh":
            x_min, y_min, width, height = self.coord
        else:
            raise ValueError(f"Unsupported coord_type: {self.coord_type}")
        return width * height

    def center(self) -> np.ndarray:
        """Calculate center point of the bounding box."""
        if self.coord_type == "xyxy":
            x_min, y_min, x_max, y_max = self.coord
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
        elif self.coord_type == "xywh":
            x_min, y_min, width, height = self.coord
            center_x = x_min + width / 2
            center_y = y_min + height / 2
        else:
            raise ValueError(f"Unsupported coord_type: {self.coord_type}")
        return np.array([center_x, center_y])
