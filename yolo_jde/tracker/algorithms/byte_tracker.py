"""ByteTracker algorithm implementation (simplified version)."""

import numpy as np
from typing import List, Dict, Any

from ..base.basetrack import BaseTrack, TrackState
from ..utils.kalman_filter import KalmanFilterXYWH
from ..utils.matching import iou_distance, linear_assignment


class BYTETrack(BaseTrack):
    """Single track for ByteTracker algorithm."""
    
    shared_kalman = KalmanFilterXYWH()
    
    def __init__(self, xywh, score, cls):
        """Initialize track."""
        super().__init__()
        
        self._xywh = np.array(xywh, dtype=np.float32)
        self.score = score
        self.cls = cls
        
        # Kalman filter state
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.tracklet_len = 0
    
    @property
    def xywh(self):
        """Get xywh coordinates."""
        if self.mean is None:
            return self._xywh
        return self.mean[:4].copy()
    
    @property
    def xyxy(self):
        """Get xyxy coordinates."""
        xywh = self.xywh
        x1 = xywh[0] - xywh[2] / 2
        y1 = xywh[1] - xywh[3] / 2
        x2 = xywh[0] + xywh[2] / 2
        y2 = xywh[1] + xywh[3] / 2
        return np.array([x1, y1, x2, y2])
    
    @property
    def tlbr(self):
        """Get tlbr coordinates (same as xyxy)."""
        return self.xyxy
    
    def predict(self):
        """Predict next state using Kalman filter."""
        if self.mean is not None and self.covariance is not None:
            self.mean, self.covariance = self.shared_kalman.predict(self.mean, self.covariance)
    
    def activate(self, kalman_filter, frame_id):
        """Activate track."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        
        # Initialize Kalman filter
        self.mean, self.covariance = self.kalman_filter.initiate(self.xywh)
        
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
    
    def re_activate(self, new_track, frame_id, new_id=False):
        """Re-activate lost track."""
        # Update Kalman filter
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, new_track.xywh)
        
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
    
    def update(self, new_track, frame_id):
        """Update track with new detection."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        
        # Update Kalman filter
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, new_track.xywh)
        
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.cls = new_track.cls


class BYTETracker:
    """ByteTracker algorithm for multi-object tracking."""
    
    def __init__(self, args):
        """Initialize ByteTracker."""
        self.tracked_stracks = []  # type: list[BYTETrack]
        self.lost_stracks = []  # type: list[BYTETrack]
        self.removed_stracks = []  # type: list[BYTETrack]
        
        self.frame_id = 0
        self.args = args
        
        # Tracking parameters
        self.track_high_thresh = args.get('track_high_thresh', 0.6)
        self.track_low_thresh = args.get('track_low_thresh', 0.1)
        self.new_track_thresh = args.get('new_track_thresh', 0.7)
        self.track_buffer = args.get('track_buffer', 30)
        self.match_thresh = args.get('match_thresh', 0.8)
        
        # Kalman filter
        self.kalman_filter = KalmanFilterXYWH()
    
    def update(self, detections: Dict[str, Any], img: np.ndarray) -> Dict[str, Any]:
        """
        Update tracker with new detections.
        
        Args:
            detections: Detection results containing boxes, scores, classes
            img: Current frame image
            
        Returns:
            Tracking results
        """
        self.frame_id += 1
        
        # Extract detection data
        boxes = detections.get('boxes', np.array([]).reshape(0, 4))  # xyxy format
        scores = detections.get('scores', np.array([]))
        classes = detections.get('classes', np.array([]))
        
        # Convert xyxy to xywh
        if len(boxes) > 0:
            xywh_boxes = np.zeros_like(boxes)
            xywh_boxes[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # center x
            xywh_boxes[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # center y
            xywh_boxes[:, 2] = boxes[:, 2] - boxes[:, 0]        # width
            xywh_boxes[:, 3] = boxes[:, 3] - boxes[:, 1]        # height
        else:
            xywh_boxes = boxes
        
        # Create detection tracks
        if len(scores) > 0:
            # Filter by confidence
            high_conf_mask = scores >= self.track_high_thresh
            low_conf_mask = (scores >= self.track_low_thresh) & (scores < self.track_high_thresh)
            
            # High confidence detections
            if np.any(high_conf_mask):
                dets_high = xywh_boxes[high_conf_mask]
                scores_high = scores[high_conf_mask]
                classes_high = classes[high_conf_mask] if len(classes) > 0 else np.zeros(len(dets_high))
            else:
                dets_high = np.array([]).reshape(0, 4)
                scores_high = np.array([])
                classes_high = np.array([])
            
            # Low confidence detections
            if np.any(low_conf_mask):
                dets_low = xywh_boxes[low_conf_mask]
                scores_low = scores[low_conf_mask]
                classes_low = classes[low_conf_mask] if len(classes) > 0 else np.zeros(len(dets_low))
            else:
                dets_low = np.array([]).reshape(0, 4)
                scores_low = np.array([])
                classes_low = np.array([])
        else:
            dets_high = dets_low = np.array([]).reshape(0, 4)
            scores_high = scores_low = np.array([])
            classes_high = classes_low = np.array([])
        
        # Create track objects
        detections_high = [BYTETrack(det, score, cls) 
                          for det, score, cls in zip(dets_high, scores_high, classes_high)]
        
        detections_low = [BYTETrack(det, score, cls) 
                         for det, score, cls in zip(dets_low, scores_low, classes_low)]
        
        # Predict existing tracks
        for track in self.tracked_stracks:
            track.predict()
        
        # First association with high confidence detections
        tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        
        # Match with high confidence detections
        matches, unmatched_tracks, unmatched_dets = self._associate(
            tracked_stracks, detections_high, self.match_thresh
        )
        
        # Update matched tracks
        for m in matches:
            track = tracked_stracks[m[0]]
            det = detections_high[m[1]]
            track.update(det, self.frame_id)
        
        # Second association with low confidence detections
        lost_stracks = [t for t in self.lost_stracks if self.frame_id - t.end_frame <= self.track_buffer]
        
        # Match unmatched tracks with low confidence detections
        if len(unmatched_tracks) > 0 and len(detections_low) > 0:
            unmatched_tracked_stracks = [tracked_stracks[i] for i in unmatched_tracks]
            matches_low, unmatched_tracks_low, unmatched_dets_low = self._associate(
                unmatched_tracked_stracks, detections_low, 0.5
            )
            
            for m in matches_low:
                track = unmatched_tracked_stracks[m[0]]
                det = detections_low[m[1]]
                track.update(det, self.frame_id)
            
            # Update unmatched tracks
            for i in unmatched_tracks_low:
                track = unmatched_tracked_stracks[i]
                track.mark_lost()
        else:
            # Mark all unmatched tracks as lost
            for i in unmatched_tracks:
                track = tracked_stracks[i]
                track.mark_lost()
        
        # Initialize new tracks
        unmatched_detections = [detections_high[i] for i in unmatched_dets]
        for det in unmatched_detections:
            if det.score >= self.new_track_thresh:
                det.activate(self.kalman_filter, self.frame_id)
                self.tracked_stracks.append(det)
        
        # Update track lists
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.lost_stracks = [t for t in self.tracked_stracks + self.lost_stracks if t.state == TrackState.Lost]
        
        # Remove old tracks
        self.lost_stracks = [t for t in self.lost_stracks if self.frame_id - t.end_frame <= self.track_buffer]
        
        # Prepare output
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        
        # Format output
        if len(output_stracks) > 0:
            output_boxes = np.array([track.xyxy for track in output_stracks])
            output_ids = np.array([track.track_id for track in output_stracks])
            output_scores = np.array([track.score for track in output_stracks])
            output_classes = np.array([track.cls for track in output_stracks])
        else:
            output_boxes = np.array([]).reshape(0, 4)
            output_ids = np.array([])
            output_scores = np.array([])
            output_classes = np.array([])
        
        return {
            'boxes': output_boxes,
            'track_ids': output_ids,
            'scores': output_scores,
            'classes': output_classes,
            'embeddings': detections.get('embeddings', np.array([]).reshape(0, 128))
        }
    
    def _associate(self, tracks, detections, thresh):
        """Associate tracks with detections using IoU."""
        if len(tracks) == 0 or len(detections) == 0:
            return np.empty((0, 2), dtype=int), list(range(len(tracks))), list(range(len(detections)))
        
        # Compute IoU cost matrix
        cost_matrix = iou_distance(tracks, detections)
        
        # Apply threshold
        cost_matrix[cost_matrix > thresh] = thresh + 1e-5
        
        # Solve assignment
        matches, unmatched_tracks, unmatched_dets = linear_assignment(cost_matrix, thresh)
        
        return matches, unmatched_tracks, unmatched_dets
