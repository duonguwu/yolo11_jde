import numpy as np
from numpy import typing as npt
from typing import Literal, Dict, List
from collections import defaultdict
from dataclasses import dataclass, field

from camera_ai_evaluator.entity.frame import FrameId
from camera_ai_evaluator.entity.object import DetectedObject, ClassId
from camera_ai_evaluator.entity.dataset import DetectedDataset
from camera_ai_evaluator.entity.bounding_box import (
    xywh_to_xyxy,
)
from camera_ai_evaluator.detection.helper import (
    calculate_iou_matrix,
    get_all_objects_by_class,
)


def cal_iou(
    gts: DetectedDataset,
    preds: DetectedDataset,
    gt_bbox_type="xywh",
    pred_bbox_type="xywh",
):
    gt_by_img = {fa.id: fa.detected_objects for fa in gts.detected_frames.values()}
    total_iou = 0
    matched_gt_count = 0

    for pred_frame in preds.detected_frames.values():
        img_id = pred_frame.id
        if img_id not in gt_by_img:
            continue

        gt_objects = [obj for obj in gt_by_img[img_id] if obj.bounding_box]
        pred_objects = [obj for obj in pred_frame.detected_objects if obj.bounding_box]

        if not gt_objects or not pred_objects:
            continue

        gt_boxes = np.array([obj.bounding_box.coord for obj in gt_objects])
        pred_boxes = np.array([obj.bounding_box.coord for obj in pred_objects])

        if gt_bbox_type == "xywh":
            gt_boxes = xywh_to_xyxy(gt_boxes)
        if pred_bbox_type == "xywh":
            pred_boxes = xywh_to_xyxy(pred_boxes)

        iou_matrix = calculate_iou_matrix(gt_boxes, pred_boxes)

        # Với mỗi GT, tìm pred có IoU cao nhất
        best_pred_ious = np.max(iou_matrix, axis=1)

        total_iou += np.sum(best_pred_ious)
        matched_gt_count += len(gt_objects)

    return total_iou / matched_gt_count if matched_gt_count > 0 else 0


def calculate_tp_fp_fn(
    gts: DetectedDataset,
    preds: DetectedDataset,
    iou_threshold: float,
) -> dict[ClassId, dict[str, int]]:
    """
    Calculate total TP, FP, FN for each class across the dataset.

    Parameters
    ----------
    gts : list[DetectionFrameAnnotation]
        Ground-truth annotations.
    preds : list[DetectionFrameAnnotation]
        Predicted annotations.
    iou_threshold : float
        IoU threshold for matching boxes.

    Returns
    -------
    dict[ClassId, dict[str, int]]
        Stats per class, including TP, FP, FN, total_gt, total_pred.
    """
    # Group objects by frame_id
    gt_by_img: dict[FrameId, list[DetectedObject]] = defaultdict(list)
    for fa in gts.detected_frames.values():
        gt_by_img[fa.id].extend(fa.detected_objects)

    pred_by_img: dict[FrameId, list[DetectedObject]] = defaultdict(list)
    for fa in preds.detected_frames.values():
        pred_by_img[fa.id].extend(fa.detected_objects)

    all_img_ids = set(gt_by_img.keys()) | set(pred_by_img.keys())
    gt_class_ids = {
        obj.class_id
        for fa in gts.detected_frames.values()
        for obj in fa.detected_objects
    }
    pred_class_ids = {
        obj.class_id
        for fa in preds.detected_frames.values()
        for obj in fa.detected_objects
    }
    all_class_ids = gt_class_ids.union(pred_class_ids)

    # Initialize stats
    stats_by_class: dict[ClassId, dict[str, int]] = {
        cid: {"TP": 0, "FP": 0, "FN": 0, "total_gt": 0, "total_pred": 0}
        for cid in all_class_ids
    }

    for img_id in all_img_ids:
        gt_in_img = gt_by_img[img_id]
        pred_in_img = pred_by_img[img_id]

        for class_id in all_class_ids:
            gt_class_objs = [o for o in gt_in_img if o.class_id == class_id]
            pred_class_objs = [o for o in pred_in_img if o.class_id == class_id]

            # Convert xywh to xyxy for IoU calculation
            gt_boxes = np.array(
                [
                    (
                        xywh_to_xyxy(o.bounding_box.coord)
                        if o.bounding_box.coord_type == "xywh"
                        else o.bounding_box.coord
                    )
                    for o in gt_class_objs
                ]
            )
            pred_boxes = np.array(
                [
                    (
                        xywh_to_xyxy(o.bounding_box.coord)
                        if o.bounding_box.coord_type == "xywh"
                        else o.bounding_box.coord
                    )
                    for o in pred_class_objs
                ]
            )

            stats = stats_by_class[class_id]
            stats["total_gt"] += len(gt_class_objs)
            stats["total_pred"] += len(pred_class_objs)

            if not gt_class_objs:
                stats["FP"] += len(pred_class_objs)
                continue
            if not pred_class_objs:
                stats["FN"] += len(gt_class_objs)
                continue

            # Sort preds by confidence
            # Important: Need to sort pred_class_objs and pred_boxes together
            sorted_indices = np.argsort([-o.confidence for o in pred_class_objs])
            pred_class_objs = [pred_class_objs[i] for i in sorted_indices]
            pred_boxes = pred_boxes[sorted_indices]

            iou_matrix = calculate_iou_matrix(gt_boxes, pred_boxes)

            # Greedy matching
            matched_gt_indices: set[int] = set()

            # For each prediction, find the best GT match
            for pred_idx in range(len(pred_class_objs)):
                gt_ious = iou_matrix[:, pred_idx]
                best_gt_idx = int(np.argmax(gt_ious))
                best_iou = gt_ious[best_gt_idx]

                if best_iou >= iou_threshold and best_gt_idx not in matched_gt_indices:
                    stats["TP"] += 1
                    matched_gt_indices.add(best_gt_idx)
                else:
                    stats["FP"] += 1

            stats["FN"] += len(gt_class_objs) - len(matched_gt_indices)

    return stats_by_class


def cal_precision(
    gt: DetectedDataset,
    pred: DetectedDataset,
    iou_threshold: float,
):
    stats = calculate_tp_fp_fn(gt, pred, iou_threshold)
    total_tp = sum(s["TP"] for s in stats.values())
    total_fp = sum(s["FP"] for s in stats.values())

    if total_tp + total_fp == 0:
        return 0.0
    return total_tp / (total_tp + total_fp)


def cal_recall(
    gt: DetectedDataset,
    pred: DetectedDataset,
    iou_threshold: float,
):
    stats = calculate_tp_fp_fn(gt, pred, iou_threshold)
    total_tp = sum(s["TP"] for s in stats.values())
    total_fn = sum(s["FN"] for s in stats.values())

    if total_tp + total_fn == 0:
        return 0.0
    return total_tp / (total_tp + total_fn)


def cal_f1(
    gt: DetectedDataset,
    pred: DetectedDataset,
    iou_threshold: float,
):
    precision = cal_precision(gt, pred, iou_threshold)
    recall = cal_recall(gt, pred, iou_threshold)

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def _voc_ap(rec: list[float], prec: list[float]) -> float:
    """
    Calculate the AP given the recall and precision array, using the
    PASCAL VOC metric.
    """
    # Thêm các điểm 0 và 1 vào đầu và cuối để tính toán
    mrec = [0.0] + rec + [1.0]
    mpre = [0.0] + prec + [0.0]

    # Làm cho đường cong precision đơn điệu không tăng (monotonically decreasing)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Tìm các chỉ số nơi recall thay đổi
    i_list = [i for i in range(1, len(mrec)) if mrec[i] != mrec[i - 1]]

    # Tính diện tích dưới đường cong (AP)
    ap = 0.0
    for i in i_list:
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def cal_AP(
    gts: DetectedDataset,
    preds: DetectedDataset,
    iou_threshold: float = 0.5,
) -> Dict[ClassId, float]:
    """
    Calculates Average Precision (AP) for each class at a given IoU threshold.

    Parameters
    ----------
    gts : DetectedDataset
        Ground-truth annotations for the entire dataset.
    preds : DetectedDataset
        Predicted annotations for the entire dataset.
    iou_threshold : float, optional
        IoU threshold for matching, by default 0.5.

    Returns
    -------
    Dict[ClassId, float]
        A dictionary mapping each class ID to its calculated AP.
    """
    # Gộp các đối tượng theo frame ID để dễ truy cập
    gt_by_img: Dict[FrameId, List[DetectedObject]] = defaultdict(list)
    for fa in gts.detected_frames.values():
        gt_by_img[fa.id].extend(fa.detected_objects)

    # Lấy tất cả các class ID có trong ground-truth
    gt_class_ids = {
        obj.class_id
        for fa in gts.detected_frames.values()
        for obj in fa.detected_objects
    }
    ap_per_class = {}

    gt_obj_by_cls = get_all_objects_by_class(gts)
    pred_obj_by_cls = get_all_objects_by_class(preds)

    for class_id in gt_class_ids:
        # Lấy tất cả gt và pred cho class hiện tại
        class_gts_count = sum(1 for obj in gt_obj_by_cls[class_id])
        class_preds = sorted(
            pred_obj_by_cls[class_id],
            key=lambda o: o.confidence,
            reverse=True,
        )

        if class_gts_count == 0:
            ap_per_class[class_id] = 0.0
            continue

        if not class_preds:
            ap_per_class[class_id] = 0.0
            continue

        # Theo dõi các gt đã được khớp trong mỗi ảnh
        # key: frame_id, value: mảng boolean có kích thước bằng số gt trong ảnh đó
        gt_matched_in_img: Dict[FrameId, np.ndarray] = {}

        tp_flags = np.zeros(len(class_preds))
        fp_flags = np.zeros(len(class_preds))

        for i, pred_obj in enumerate(class_preds):
            frame_id = pred_obj.frame_id
            gts_in_frame = [
                obj for obj in gt_by_img[frame_id] if obj.class_id == class_id
            ]

            if not gts_in_frame:
                fp_flags[i] = 1
                continue

            if frame_id not in gt_matched_in_img:
                gt_matched_in_img[frame_id] = np.zeros(len(gts_in_frame), dtype=bool)

            gt_boxes = np.array(
                [
                    (
                        xywh_to_xyxy(o.bounding_box.coord)
                        if o.bounding_box.coord_type == "xywh"
                        else o.bounding_box.coord
                    )
                    for o in gts_in_frame
                ]
            )
            pred_box = np.array(
                [
                    (
                        xywh_to_xyxy(pred_obj.bounding_box.coord)
                        if pred_obj.bounding_box.coord_type == "xywh"
                        else pred_obj.bounding_box.coord
                    )
                ]
            )

            iou_matrix = calculate_iou_matrix(gt_boxes, pred_box)
            best_gt_idx = np.argmax(iou_matrix)
            best_iou = iou_matrix[best_gt_idx, 0]

            if best_iou >= iou_threshold:
                if not gt_matched_in_img[frame_id][best_gt_idx]:
                    tp_flags[i] = 1
                    gt_matched_in_img[frame_id][best_gt_idx] = True
                else:
                    fp_flags[i] = 1  # Khớp với gt đã được dùng -> FP
            else:
                fp_flags[i] = 1

        # Tính toán precision và recall tích lũy
        tp_cumsum = np.cumsum(tp_flags)
        fp_cumsum = np.cumsum(fp_flags)

        recall = tp_cumsum / class_gts_count
        precision = tp_cumsum / (
            tp_cumsum + fp_cumsum + np.finfo(float).eps
        )  # tránh chia cho 0

        ap = _voc_ap(recall.tolist(), precision.tolist())
        ap_per_class[class_id] = ap

    return ap_per_class


def cal_mAP_IoU_threshold(
    gts: DetectedDataset,
    preds: DetectedDataset,
    iou_threshold: float,
) -> float:
    """
    Calculates mean Average Precision (mAP) at a specific IoU threshold.

    Parameters
    ----------
    gts : DetectedDataset
        Ground-truth annotations.
    preds : DetectedDataset
        Predicted annotations.
    iou_threshold : float
        The IoU threshold to use for calculation.

    Returns
    -------
    float
        The mAP score.
    """
    ap_per_class = cal_AP(gts, preds, iou_threshold)

    if not ap_per_class:
        return 0.0

    return float(np.mean(list(ap_per_class.values())))


def cal_mAP(
    gts: DetectedDataset,
    preds: DetectedDataset,
) -> float:
    """
    Calculates the COCO-style mAP, which is the average of mAP scores
    over multiple IoU thresholds [0.5, 0.55, ..., 0.95].

    Parameters
    ----------
    gts : DetectedDataset
        Ground-truth annotations.
    preds : DetectedDataset
        Predicted annotations.

    Returns
    -------
    float
        The final COCO-style mAP score.
    """
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    map_scores = []

    for iou in iou_thresholds:
        map_at_iou = cal_mAP_IoU_threshold(gts, preds, float(iou))
        map_scores.append(map_at_iou)

    if not map_scores:
        return 0.0

    return float(np.mean(map_scores))


@dataclass
class MatchResult:
    # Tuple: (gt_object, pred_object, iou_score)
    true_positives: list[tuple[DetectedObject, DetectedObject, float]] = field(
        default_factory=list
    )
    false_positives: list[DetectedObject] = field(default_factory=list)
    false_negatives: list[DetectedObject] = field(default_factory=list)
    total_gt_by_class: dict[ClassId, int] = field(
        default_factory=lambda: defaultdict(int)
    )


def extract_boxes(objs: list[DetectedObject], to_xyxy: bool = True) -> np.ndarray:
    """
    Trích xuất danh sách bounding boxes từ danh sách đối tượng.
    Nếu coord_type là 'xywh' thì tự động chuyển sang xyxy.
    """
    boxes = []
    for o in objs:
        coord = o.bounding_box.coord
        if to_xyxy and o.bounding_box.coord_type == "xywh":
            coord = xywh_to_xyxy(coord)
        boxes.append(coord)
    return np.stack(boxes) if boxes else np.empty((0, 4), dtype=np.float32)


def analyze_matches_detailed(
    gts: DetectedDataset,
    preds: DetectedDataset,
    iou_threshold: float,
) -> MatchResult:
    """
    Phân tích chi tiết các cặp match, trả về danh sách các object TP, FP, FN.
    """
    result = MatchResult()

    gt_by_img: dict[FrameId, list[DetectedObject]] = defaultdict(list)
    for fa in gts.detected_frames.values():
        gt_by_img[fa.id].extend(fa.detected_objects)

    pred_by_img: dict[FrameId, list[DetectedObject]] = defaultdict(list)
    for fa in preds.detected_frames.values():
        pred_by_img[fa.id].extend(fa.detected_objects)

    all_class_ids = {
        o.class_id for fa in gts.detected_frames.values() for o in fa.detected_objects
    }

    # Đếm tổng số GT cho mỗi class
    for fa in gts.detected_frames.values():
        for obj in fa.detected_objects:
            result.total_gt_by_class[obj.class_id] += 1

    all_img_ids = set(gt_by_img.keys()) | set(pred_by_img.keys())

    for img_id in all_img_ids:
        gt_in_img = gt_by_img[img_id]
        pred_in_img = pred_by_img[img_id]

        for class_id in all_class_ids:
            gt_class_objs = [o for o in gt_in_img if o.class_id == class_id]
            pred_class_objs = [o for o in pred_in_img if o.class_id == class_id]

            if not gt_class_objs and not pred_class_objs:
                continue

            if not pred_class_objs:
                # Tất cả GT đều không được phát hiện -> FN
                result.false_negatives.extend(gt_class_objs)
                continue

            if not gt_class_objs:
                # Tất cả pred đều không có GT tương ứng -> FP
                result.false_positives.extend(pred_class_objs)
                continue

            # --- Logic Matching ---
            gt_boxes = extract_boxes(gt_class_objs)
            pred_boxes = extract_boxes(pred_class_objs)

            # Sắp xếp pred theo confidence giảm dần
            sorted_indices = np.argsort([-o.confidence for o in pred_class_objs])
            pred_class_objs = [pred_class_objs[i] for i in sorted_indices]
            pred_boxes = pred_boxes[sorted_indices]

            iou_matrix = calculate_iou_matrix(gt_boxes, pred_boxes)

            gt_matched_flags = np.zeros(len(gt_class_objs), dtype=bool)

            # Duyệt qua từng prediction
            for pred_idx, pred_obj in enumerate(pred_class_objs):
                gt_ious = iou_matrix[:, pred_idx]
                best_gt_idx = int(np.argmax(gt_ious))
                best_iou = gt_ious[best_gt_idx]

                if best_iou >= iou_threshold and not gt_matched_flags[best_gt_idx]:
                    # Match thành công (TP)
                    gt_obj = gt_class_objs[best_gt_idx]
                    result.true_positives.append((gt_obj, pred_obj, best_iou))
                    gt_matched_flags[best_gt_idx] = True
                else:
                    # Match thất bại (FP)
                    result.false_positives.append(pred_obj)

            # Các GT không được match là FN
            for gt_idx, is_matched in enumerate(gt_matched_flags):
                if not is_matched:
                    result.false_negatives.append(gt_class_objs[gt_idx])

    return result
