"""Matching utilities for object tracking."""

import numpy as np
import scipy.spatial.distance as dist
from typing import List, Tuple, Optional
import lapx


def embedding_distance(tracks: List, detections: List, metric: str = 'cosine') -> np.ndarray:
    """
    Compute embedding distance between tracks and detections using ReID features.
    
    Args:
        tracks: List of track objects with smooth_feat or curr_feature attribute
        detections: List of detection objects with curr_feat or feature attribute  
        metric: Distance metric ('cosine', 'euclidean')
        
    Returns:
        Distance matrix of shape (len(tracks), len(detections))
    """
    if len(tracks) == 0 or len(detections) == 0:
        return np.empty((len(tracks), len(detections)))
    
    # Extract features from tracks
    track_features = []
    for track in tracks:
        if hasattr(track, 'smooth_feat') and track.smooth_feat is not None:
            track_features.append(track.smooth_feat)
        elif hasattr(track, 'curr_feature') and track.curr_feature is not None:
            track_features.append(track.curr_feature)
        elif hasattr(track, 'features') and len(track.features) > 0:
            track_features.append(track.features[-1])
        else:
            # Use zero vector as fallback
            track_features.append(np.zeros(128))
    
    # Extract features from detections
    det_features = []
    for det in detections:
        if hasattr(det, 'curr_feat') and det.curr_feat is not None:
            det_features.append(det.curr_feat)
        elif hasattr(det, 'feature') and det.feature is not None:
            det_features.append(det.feature)
        else:
            # Use zero vector as fallback
            det_features.append(np.zeros(128))
    
    # Compute distance matrix
    track_features = np.array(track_features)
    det_features = np.array(det_features)
    
    if metric == 'cosine':
        # Normalize features for cosine distance
        track_features = track_features / (np.linalg.norm(track_features, axis=1, keepdims=True) + 1e-8)
        det_features = det_features / (np.linalg.norm(det_features, axis=1, keepdims=True) + 1e-8)
        
    distance_matrix = dist.cdist(track_features, det_features, metric=metric)
    return distance_matrix


def iou_distance(tracks: List, detections: List) -> np.ndarray:
    """
    Compute IoU distance between tracks and detections.
    
    Args:
        tracks: List of track objects with tlbr or xyxy bbox attribute
        detections: List of detection objects with tlbr or xyxy bbox attribute
        
    Returns:
        IoU distance matrix (1 - IoU) of shape (len(tracks), len(detections))
    """
    if len(tracks) == 0 or len(detections) == 0:
        return np.empty((len(tracks), len(detections)))
    
    # Extract bounding boxes
    track_boxes = []
    for track in tracks:
        if hasattr(track, 'tlbr'):
            track_boxes.append(track.tlbr)
        elif hasattr(track, 'xyxy'):
            track_boxes.append(track.xyxy)
        elif hasattr(track, 'xywh'):
            # Convert xywh to xyxy
            x, y, w, h = track.xywh
            track_boxes.append([x - w/2, y - h/2, x + w/2, y + h/2])
        else:
            track_boxes.append([0, 0, 1, 1])  # fallback
    
    det_boxes = []
    for det in detections:
        if hasattr(det, 'tlbr'):
            det_boxes.append(det.tlbr)
        elif hasattr(det, 'xyxy'):
            det_boxes.append(det.xyxy)
        elif hasattr(det, 'xywh'):
            # Convert xywh to xyxy
            x, y, w, h = det.xywh
            det_boxes.append([x - w/2, y - h/2, x + w/2, y + h/2])
        else:
            det_boxes.append([0, 0, 1, 1])  # fallback
    
    track_boxes = np.array(track_boxes)
    det_boxes = np.array(det_boxes)
    
    # Compute IoU matrix
    ious = bbox_iou_batch(track_boxes, det_boxes)
    
    # Return distance (1 - IoU)
    return 1 - ious


def bbox_iou_batch(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU between two sets of bounding boxes.
    
    Args:
        boxes1: Array of shape (N, 4) in format [x1, y1, x2, y2]
        boxes2: Array of shape (M, 4) in format [x1, y1, x2, y2]
        
    Returns:
        IoU matrix of shape (N, M)
    """
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]))
    
    # Expand dimensions for broadcasting
    boxes1 = boxes1[:, None, :]  # (N, 1, 4)
    boxes2 = boxes2[None, :, :]  # (1, M, 4)
    
    # Compute intersection
    lt = np.maximum(boxes1[..., :2], boxes2[..., :2])  # left-top
    rb = np.minimum(boxes1[..., 2:], boxes2[..., 2:])  # right-bottom
    
    wh = np.maximum(0, rb - lt)  # width-height
    intersection = wh[..., 0] * wh[..., 1]
    
    # Compute areas
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    
    # Compute IoU
    union = area1 + area2 - intersection
    iou = intersection / np.maximum(union, 1e-8)
    
    return iou


def fuse_motion(kf, cost_matrix: np.ndarray, tracks: List, detections: List, 
               only_position: bool = False, lambda_: float = 0.98) -> np.ndarray:
    """
    Fuse motion information with appearance cost matrix.
    
    Args:
        kf: Kalman filter instance
        cost_matrix: Appearance cost matrix
        tracks: List of track objects
        detections: List of detection objects
        only_position: Whether to use only position for gating
        lambda_: Fusion weight (higher = more appearance, lower = more motion)
        
    Returns:
        Fused cost matrix with gating applied
    """
    if len(tracks) == 0 or len(detections) == 0:
        return cost_matrix
    
    gating_dim = 2 if only_position else 4
    gating_threshold = 9.4877 if gating_dim == 4 else 5.9915  # chi2inv95[gating_dim]
    
    measurements = []
    for det in detections:
        if hasattr(det, 'to_xyah'):
            measurements.append(det.to_xyah())
        elif hasattr(det, 'xywh'):
            x, y, w, h = det.xywh
            measurements.append([x, y, w/h, h])  # convert to xyah
        else:
            measurements.append([0, 0, 1, 1])  # fallback
    
    measurements = np.array(measurements)
    
    for row, track in enumerate(tracks):
        if hasattr(track, 'mean') and hasattr(track, 'covariance'):
            gating_distance = kf.gating_distance(
                track.mean, track.covariance, measurements, only_position, metric='maha'
            )
            # Apply gating: set cost to infinity for distant associations
            cost_matrix[row, gating_distance > gating_threshold] = np.inf
            
            # Fuse motion cost with appearance cost
            motion_cost = gating_distance / gating_threshold
            cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * motion_cost.flatten()
    
    return cost_matrix


def linear_assignment(cost_matrix: np.ndarray, thresh: float = np.inf) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve linear assignment problem using Hungarian algorithm.
    
    Args:
        cost_matrix: Cost matrix of shape (N, M)
        thresh: Threshold for valid assignments
        
    Returns:
        Tuple of (matches, unmatched_tracks, unmatched_detections)
        - matches: Array of shape (K, 2) with matched (track_idx, det_idx) pairs
        - unmatched_tracks: Array of unmatched track indices
        - unmatched_detections: Array of unmatched detection indices
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0]),
            np.arange(cost_matrix.shape[1])
        )
    
    # Solve assignment problem
    try:
        # Use lapx for Hungarian algorithm (faster than scipy)
        _, x, y = lapx.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
        matches = np.array(matches)
    except ImportError:
        # Fallback to scipy if lapx not available
        from scipy.optimize import linear_sum_assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        matches = []
        for r, c in zip(row_indices, col_indices):
            if cost_matrix[r, c] <= thresh:
                matches.append([r, c])
        matches = np.array(matches)
    
    # Find unmatched tracks and detections
    if matches.size > 0:
        matched_tracks = matches[:, 0]
        matched_dets = matches[:, 1]
    else:
        matched_tracks = np.array([])
        matched_dets = np.array([])
    
    unmatched_tracks = np.array([i for i in range(cost_matrix.shape[0]) if i not in matched_tracks])
    unmatched_dets = np.array([i for i in range(cost_matrix.shape[1]) if i not in matched_dets])
    
    return matches, unmatched_tracks, unmatched_dets
