import numpy as np
from camera_ai_evaluator.tracking.entity.metric_input import TrackingMetricInput
from camera_ai_evaluator.tracking.metrics import (
    HOTA,
)  # sau khi bạn lưu lại class này

# giả lập với dữ liệu dạng TrackingMetricInput
data = TrackingMetricInput(
    gt_ids=[np.array([0, 1]), np.array([0, 1])],
    tracker_ids=[np.array([0, 1]), np.array([0, 1])],
    similarity_scores=[
        np.array([[0.9, 0.1], [0.1, 0.8]]),
        np.array([[0.8, 0.2], [0.2, 0.7]]),
    ],
    num_gt_dets=4,
    num_pred_dets=4,
    num_gt_ids=2,
    num_pred_ids=2,
)
metric = HOTA()
res = metric.eval_sequence(data)

# check kết quả có đúng không
