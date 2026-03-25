import numpy as np
from collections import defaultdict

from camera_ai_evaluator.entity.object import DetectedObject, ClassId

from camera_ai_evaluator.entity.dataset import DetectedDataset


def calculate_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Tính toán ma trận IoU giữa hai tập bounding box sử dụng numpy

    Args:
        boxes1 (np.ndarray): Mảng numpy có shape (N, 4) chứa N bounding box.
        boxes2 (np.ndarray): Mảng numpy có shape (M, 4) chứa M bounding box.

    Returns:
        np.ndarray: Ma trận IoU có shape (N, M).
    """
    # Tính diện tích của các box
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Tìm tọa độ của vùng giao nhau (intersection)
    # Kỹ thuật broadcasting của numpy: (N, 1, 2) và (M, 2) -> (N, M, 2)
    inter_top_left = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    inter_bottom_right = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])

    # Tính width và height của vùng giao nhau
    inter_wh = np.maximum(0.0, inter_bottom_right - inter_top_left)

    # Tính diện tích vùng giao nhau
    inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]

    # Tính diện tích vùng hợp nhất (union)
    # Kỹ thuật broadcasting: (N, 1) + (M,) - (N, M) -> (N, M)
    union_area = area1[:, None] + area2 - inter_area

    # Tính IoU, tránh trường hợp chia cho 0
    iou = inter_area / np.maximum(union_area, 1e-8)

    return iou


def get_all_objects_by_class(
    anotations: DetectedDataset,
) -> dict[ClassId, list[DetectedObject]]:
    """
    Tối ưu việc trích xuất và gom nhóm các object theo class_id từ toàn bộ collection.
    Điều này tránh việc phải lặp qua toàn bộ dữ liệu nhiều lần.
    """
    objects_by_class: dict[ClassId, list[DetectedObject]] = defaultdict(list)
    for frame_anot in anotations.detected_frames.values():
        for obj in frame_anot.detected_objects:
            objects_by_class[obj.class_id].append(obj)
    return objects_by_class
