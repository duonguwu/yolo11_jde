import numpy as np
import cv2
from typing import Literal


def draw_box(
    image: np.ndarray,
    box: np.ndarray,
    label: str,
    color: tuple[int, int, int],
    box_type: Literal["xyxy", "xywh"] = "xywh",
    label_position: Literal[
        "top_left", "top_right", "bottom_left", "bottom_right", "center"
    ] = "top_left",
    text_shadow: bool = True,
) -> np.ndarray:
    """
    Draw bounding box and its label.

    Args:
        image (np.ndarray): Input image.
        box (np.ndarray): Bounding box coordinates.
        label (str): Label text.
        color (tuple[int, int, int]): Box color in BGR format.
        box_type (Literal["xyxy", "xywh"], optional): Format of the box.
        label_position (Literal["top_left", "top_right", "bottom_left", "bottom_right"]).
        text_shadow (bool): Add black outline for readability.

    Returns:
        np.ndarray: Image with box and label drawn.
    """

    # Parse bbox
    if box_type == "xywh":
        x, y, w, h = map(int, box)
        x1, y1, x2, y2 = x, y, x + w, y + h
    else:
        x1, y1, x2, y2 = map(int, box)

    # Draw main box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # Label text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1

    label_size, base_line = cv2.getTextSize(label, font, scale, thickness)
    text_w, text_h = label_size
    pad = 4

    # Compute position inside box
    if label_position == "top_left":
        text_x = x1 + pad
        text_y = y1 + text_h + pad
    elif label_position == "top_right":
        text_x = x2 - text_w - pad
        text_y = y1 + text_h + pad
    elif label_position == "bottom_left":
        text_x = x1 + pad
        text_y = y2 - pad
    elif label_position == "bottom_right":
        text_x = x2 - text_w - pad
        text_y = y2 - pad
    elif label_position == "center":
        text_x = x1 + (x2 - x1 - text_w) // 2
        text_y = y1 + (y2 - y1 - text_h) // 2
    else:
        raise ValueError(f"Invalid label_position: {label_position}")

    # Clamp inside box
    text_x = max(x1 + pad, min(text_x, x2 - text_w - pad))
    text_y = max(y1 + text_h + pad, min(text_y, y2 - pad))

    # Background rectangle
    box_start = (text_x - pad, text_y - text_h - pad)
    box_end = (text_x + text_w + pad, text_y + base_line)

    cv2.rectangle(image, box_start, box_end, color, cv2.FILLED)

    # Optional text shadow (stroke effect)
    if text_shadow:
        for dx in (-1, 1):
            for dy in (-1, 1):
                cv2.putText(
                    image,
                    label,
                    (text_x + dx, text_y + dy),
                    font,
                    scale,
                    (0, 0, 0),
                    thickness + 1,
                    cv2.LINE_AA,
                )

    # Main text
    cv2.putText(
        image,
        label,
        (text_x, text_y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    return image
