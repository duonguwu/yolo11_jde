import copy
from dataclasses import dataclass
from typing import Any, Callable, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment

from camera_ai_evaluator.entity.track import Tracking, TrackId, SingleTrack
from camera_ai_evaluator.entity.frame import FrameId, DetectedFrame, Frame
from camera_ai_evaluator.entity.object import DetectedObject, ObjectId, ClassId
from camera_ai_evaluator.entity.bounding_box import BoundingBox


@dataclass
class TrackingMetricInput:
    gt_ids: list[np.ndarray]
    tracker_ids: list[np.ndarray]
    similarity_scores: list[np.ndarray]
    num_gt_dets: int
    num_pred_dets: int
    num_gt_ids: int
    num_pred_ids: int


class MetricInputConverter:
    """
    onvert TrackingResult into a standardized metric input format.
    This format can be consumed by evaluation function.
    """

    @staticmethod
    def _interpolate_tracking_data(
        tracking_data: Tracking, all_frame_ids: list[FrameId]
    ) -> Tracking:
        """
        Interpolate missing frames for each track in the tracking data.
        """
        # Create a deep copy to avoid modifying the original data
        interp_tracking = copy.deepcopy(tracking_data)

        # Create detected frames for missing frame IDs
        for frame_id in all_frame_ids:
            if frame_id not in interp_tracking.frames:
                # Lấy thông tin width/height từ frame đầu tiên có thể
                base_frame_info = next(iter(tracking_data.frames.values()), None)
                if base_frame_info:
                    interp_tracking.frames[frame_id] = DetectedFrame(
                        id=frame_id,
                        width=base_frame_info.width,
                        height=base_frame_info.height,
                        detected_objects=[],
                        image_name="",
                        video_name=base_frame_info.video_name,
                    )

        for track_id, track in tracking_data.tracks.items():
            if len(track.frame_indices) < 2:
                continue

            # Sắp xếp để đảm bảo thứ tự frame là đúng
            sorted_indices = np.argsort(track.frame_indices)
            sorted_frames = np.array(track.frame_indices)[sorted_indices]
            sorted_bboxes = np.array(track.bboxes)[sorted_indices]

            # Lấy các thuộc tính khác, xử lý trường hợp không có
            sorted_confs = (
                np.array(track.confidences)[sorted_indices]
                if track.confidences
                else None
            )
            sorted_labels = (
                np.array(track.labels)[sorted_indices] if track.labels else None
            )

            # Duyệt qua tất cả các frame cần có
            for i, target_frame_id in enumerate(all_frame_ids):
                # Nếu frame đã tồn tại trong track, bỏ qua
                if target_frame_id in sorted_frames:
                    continue

                # Tìm frame trước đó (prev) và frame kế tiếp (next)
                next_idx = np.searchsorted(sorted_frames, target_frame_id)

                # Không thể nội suy nếu không có frame ở cả hai phía
                if next_idx == 0 or next_idx == len(sorted_frames):
                    continue

                prev_idx = next_idx - 1

                prev_frame = sorted_frames[prev_idx]
                next_frame = sorted_frames[next_idx]

                # Tính toán tỉ lệ nội suy
                ratio = (target_frame_id - prev_frame) / (next_frame - prev_frame)

                # Nội suy bounding box
                prev_bbox = sorted_bboxes[prev_idx].coord
                next_bbox = sorted_bboxes[next_idx].coord
                interp_coord = prev_bbox + ratio * (next_bbox - prev_bbox)
                interp_bbox = BoundingBox(coord=interp_coord)

                # Lấy các thuộc tính khác (thường lấy của frame trước đó)
                confidence = (
                    sorted_confs[prev_idx] if sorted_confs is not None else 1.0
                )  # GT confidence is 1.0
                label = sorted_labels[prev_idx] if sorted_labels is not None else None
                # class_id của object phải lấy từ một object mẫu
                class_id_sample = interp_tracking.tracks[track_id].labels[0]
                if class_id_sample is None:
                    class_id_sample = 0  # Mặc định nếu không có

                # Tạo đối tượng DetectedObject mới
                interp_obj = DetectedObject(
                    id=ObjectId(track_id),
                    frame_id=target_frame_id,
                    class_id=ClassId(class_id_sample),
                    confidence=confidence,
                    bounding_box=interp_bbox,
                    label=label,
                )

                # Thêm đối tượng đã nội suy vào frame tương ứng
                if target_frame_id in interp_tracking.frames:
                    interp_tracking.frames[target_frame_id].add_object(interp_obj)

        return interp_tracking

    @staticmethod
    def from_tracking_result(
        tracking_gt: Tracking,
        tracking_result: Tracking,
        similarity_func: Callable,
        apply_consider_masks: bool = False,
        is_interpolate: bool = False,
    ) -> TrackingMetricInput:
        """
        Convert tracking and groundtruth result to tracking metric input

        Args:
            tracking_gt (Tracking): _description_
            tracking_result (Tracking): _description_
            similarity_func (Callable): _description_

        Returns:
            TrackingMetricInput: _description_
        """
        gt_frame_keys = set(tracking_gt.frames.keys())
        tr_frame_keys = set(tracking_result.frames.keys())

        if is_interpolate:
            # Hợp nhất và sắp xếp tất cả các frame_id từ cả hai nguồn
            all_frame_ids = sorted(list(gt_frame_keys.union(tr_frame_keys)))
            # Thực hiện nội suy
            tracking_gt = MetricInputConverter._interpolate_tracking_data(
                tracking_gt, all_frame_ids
            )
            tracking_result = MetricInputConverter._interpolate_tracking_data(
                tracking_result, all_frame_ids
            )
        else:
            # Chỉ sử dụng các frame có trong GT (hành vi tiêu chuẩn)
            all_frame_ids = sorted(list(gt_frame_keys))

        gt_ids_list, tracker_ids_list, sims_list = [], [], []
        unique_gt_ids, unique_tr_ids = [], []
        num_gt_dets = 0
        num_tr_dets = 0

        for frame_id in all_frame_ids:
            gt_objs = tracking_gt.frames.get(frame_id)
            tr_objs = tracking_result.frames.get(frame_id)

            if gt_objs is None or tr_objs is None:
                continue

            gt_ids = gt_objs.get_obj_ids()
            tr_ids = tr_objs.get_obj_ids()
            gt_dets = gt_objs.get_obj_bboxes()
            tr_dets = tr_objs.get_obj_bboxes()

            # ================= LOGIC HOÀN THIỆN BẮT ĐẦU TẠI ĐÂY =================
            # apply consider masks
            # chỉ giữ lại các đối tượng có tâm nằm trong vùng `consider_masks`
            if apply_consider_masks and tracking_gt.consider_masks is not None:
                mask = tracking_gt.consider_masks

                # Lọc các đối tượng ground-truth
                if len(gt_dets) > 0:
                    # Lấy tâm của tất cả các bounding box
                    gt_centers = np.array(
                        [obj.bounding_box.center() for obj in gt_objs.detected_objects]
                    )
                    # Chuyển tọa độ tâm thành số nguyên để làm chỉ số cho mask
                    gt_centers_int = gt_centers.astype(int)

                    # Giới hạn tọa độ để không vượt ra ngoài kích thước của mask
                    gt_centers_int[:, 0] = np.clip(
                        gt_centers_int[:, 0], 0, mask.shape[1] - 1
                    )
                    gt_centers_int[:, 1] = np.clip(
                        gt_centers_int[:, 1], 0, mask.shape[0] - 1
                    )

                    # Tạo một boolean mask để lọc: True nếu tâm nằm trong vùng hợp lệ
                    gt_keep_mask = mask[
                        gt_centers_int[:, 1], gt_centers_int[:, 0]
                    ].astype(bool)

                    # Áp dụng mask để lọc lại danh sách
                    gt_ids = gt_ids[gt_keep_mask]
                    gt_dets = gt_dets[gt_keep_mask]

                # Lọc các đối tượng của tracker (tương tự)
                if len(tr_dets) > 0:
                    tr_centers = np.array(
                        [obj.bounding_box.center() for obj in tr_objs.detected_objects]
                    )
                    tr_centers_int = tr_centers.astype(int)

                    tr_centers_int[:, 0] = np.clip(
                        tr_centers_int[:, 0], 0, mask.shape[1] - 1
                    )
                    tr_centers_int[:, 1] = np.clip(
                        tr_centers_int[:, 1], 0, mask.shape[0] - 1
                    )

                    tr_keep_mask = mask[
                        tr_centers_int[:, 1], tr_centers_int[:, 0]
                    ].astype(bool)

                    tr_ids = tr_ids[tr_keep_mask]
                    tr_dets = tr_dets[tr_keep_mask]
            # ================= LOGIC HOÀN THIỆN KẾT THÚC TẠI ĐÂY =================

            # Tính similarity
            if len(gt_dets) > 0 and len(tr_dets) > 0:
                sim = similarity_func(gt_dets, tr_dets, is_encoded=True, do_ioa=False)
            else:
                sim = np.zeros((len(gt_dets), len(tr_dets)))

            gt_ids_list.append(gt_ids)
            tracker_ids_list.append(tr_ids)
            sims_list.append(sim)

            num_gt_dets += len(gt_ids)
            num_tr_dets += len(tr_ids)
            unique_gt_ids += list(np.unique(gt_ids))
            unique_tr_ids += list(np.unique(tr_ids))

        # Chuẩn hóa lại id để liên tục
        if not unique_gt_ids:  # Tránh lỗi nếu không có đối tượng GT nào
            unique_gt_ids = np.array([])
        else:
            unique_gt_ids = np.unique(unique_gt_ids)

        if not unique_tr_ids:  # Tránh lỗi nếu không có đối tượng tracker nào
            unique_tr_ids = np.array([])
        else:
            unique_tr_ids = np.unique(unique_tr_ids)

        gt_id_map = (
            np.full(np.max(unique_gt_ids) + 1, np.nan)
            if len(unique_gt_ids) > 0
            else np.array([])
        )
        if len(unique_gt_ids) > 0:
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))

        tr_id_map = (
            np.full(np.max(unique_tr_ids) + 1, np.nan)
            if len(unique_tr_ids) > 0
            else np.array([])
        )
        if len(unique_tr_ids) > 0:
            tr_id_map[unique_tr_ids] = np.arange(len(unique_tr_ids))

        for frame_idx in range(len(gt_ids_list)):
            if len(gt_ids_list[frame_idx]) > 0:
                gt_ids_list[frame_idx] = gt_id_map[gt_ids_list[frame_idx]].astype(int)
            if len(tracker_ids_list[frame_idx]) > 0:
                tracker_ids_list[frame_idx] = tr_id_map[
                    tracker_ids_list[frame_idx]
                ].astype(int)

        return TrackingMetricInput(
            gt_ids=gt_ids_list,
            tracker_ids=tracker_ids_list,
            similarity_scores=sims_list,
            num_gt_dets=num_gt_dets,
            num_pred_dets=num_tr_dets,
            num_gt_ids=len(unique_gt_ids),
            num_pred_ids=len(unique_tr_ids),
        )
