"""SMILEtrack algorithm implementation."""

import numpy as np
from typing import List, Dict, Any, Optional

from ..base.basetrack import BaseTrack, TrackState
from ..utils.kalman_filter import KalmanFilterXYAH
from ..utils.matching import embedding_distance, iou_distance, fuse_motion, linear_assignment


class STrack(BaseTrack):
    """Single track for SMILEtrack algorithm."""
    
    shared_kalman = KalmanFilterXYAH()
    
    def __init__(self, xywh, score, cls, feat=None):
        """Initialize track."""
        super().__init__()
        
        # Convert xywh to xyah (x, y, aspect_ratio, height)
        self._xywh = np.array(xywh, dtype=np.float32)
        self.score = score
        self.cls = cls
        self.curr_feature = feat
        self.smooth_feat = feat.copy() if feat is not None else None
        self.alpha = 0.9  # Feature smoothing factor
        
        # Kalman filter state
        self.mean, self.covariance = None, None
        self.is_activated = False
        
    @property
    def xywh(self):
        """Get xywh coordinates."""
        if self.mean is None:
            return self._xywh
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]  # convert aspect ratio back to width
        return ret
    
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
    
    def to_xyah(self):
        """Convert to xyah format for Kalman filter."""
        xywh = self.xywh
        return np.array([xywh[0], xywh[1], xywh[2] / xywh[3], xywh[3]])
    
    def predict(self):
        """Predict next state using Kalman filter."""
        if self.mean is not None and self.covariance is not None:
            self.mean, self.covariance = self.shared_kalman.predict(self.mean, self.covariance)
    
    def activate(self, kalman_filter, frame_id):
        """Activate track."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        
        # Initialize Kalman filter
        xyah = self.to_xyah()
        self.mean, self.covariance = self.kalman_filter.initiate(xyah)
        
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
    
    def re_activate(self, new_track, frame_id, new_id=False):
        """Re-activate lost track."""
        # Update Kalman filter
        xyah = new_track.to_xyah()
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, xyah)
        
        # Update features
        if new_track.curr_feature is not None:
            self.update_features(new_track.curr_feature)
        
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
        xyah = new_track.to_xyah()
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, xyah)
        
        # Update features
        if new_track.curr_feature is not None:
            self.update_features(new_track.curr_feature)
        
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.cls = new_track.cls
    
    def update_features(self, feat):
        """Update features with exponential moving average."""
        if feat is None:
            return
        
        feat = feat / (np.linalg.norm(feat) + 1e-8)  # L2 normalize
        
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
            self.smooth_feat = self.smooth_feat / (np.linalg.norm(self.smooth_feat) + 1e-8)
        
        self.curr_feature = feat
        self.features.append(feat)
    
    @staticmethod
    def multi_predict(stracks):
        """Predict multiple tracks."""
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov


class SMILEtrack:
    """SMILEtrack algorithm for multi-object tracking."""
    
    def __init__(self, args, frame_rate: float = 30):
        """Initialize SMILEtrack."""
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        
        self.frame_id = 0
        self.args = args
        self.frame_rate = frame_rate
        
        # Tracking parameters
        self.track_high_thresh = args.get('track_high_thresh', 0.6)
        self.track_low_thresh = args.get('track_low_thresh', 0.1)
        self.new_track_thresh = args.get('new_track_thresh', 0.7)
        self.track_buffer = args.get('track_buffer', 30)
        self.match_thresh = args.get('match_thresh', 0.8)
        self.with_reid = args.get('with_reid', True)
        self.proximity_thresh = args.get('proximity_thresh', 0.5)
        self.appearance_thresh = args.get('appearance_thresh', 0.25)
        self.method = args.get('method', 1)
        self.buffer_size = int(self.frame_rate / 30.0 * self.track_buffer)
        self.max_time_lost = self.buffer_size
        
        # Kalman filter
        self.kalman_filter = KalmanFilterXYAH()
    
    def update(self, detections: Dict[str, Any], img: np.ndarray) -> Dict[str, Any]:
        """
        Update tracker with new detections.
        
        Args:
            detections: Detection results containing boxes, scores, classes, embeddings
            img: Current frame image
            
        Returns:
            Tracking results
        """
        self.frame_id += 1
        
        # Extract detection data
        boxes = detections.get('boxes', np.array([]).reshape(0, 4))  # xyxy format
        scores = detections.get('scores', np.array([]))
        classes = detections.get('classes', np.array([]))
        embeddings = detections.get('embeddings', np.array([]).reshape(0, 128))
        
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
                embeds_high = embeddings[high_conf_mask] if len(embeddings) > 0 else None
            else:
                dets_high = np.array([]).reshape(0, 4)
                scores_high = np.array([])
                classes_high = np.array([])
                embeds_high = None
            
            # Low confidence detections
            if np.any(low_conf_mask):
                dets_low = xywh_boxes[low_conf_mask]
                scores_low = scores[low_conf_mask]
                classes_low = classes[low_conf_mask] if len(classes) > 0 else np.zeros(len(dets_low))
                embeds_low = embeddings[low_conf_mask] if len(embeddings) > 0 else None
            else:
                dets_low = np.array([]).reshape(0, 4)
                scores_low = np.array([])
                classes_low = np.array([])
                embeds_low = None
        else:
            dets_high = dets_low = np.array([]).reshape(0, 4)
            scores_high = scores_low = np.array([])
            classes_high = classes_low = np.array([])
            embeds_high = embeds_low = None
        
        # Create track objects
        detections_high = [STrack(det, score, cls, feat) 
                          for det, score, cls, feat in zip(
                              dets_high, scores_high, classes_high,
                              embeds_high if embeds_high is not None else [None] * len(dets_high)
                          )]
        
        detections_low = [STrack(det, score, cls, feat) 
                         for det, score, cls, feat in zip(
                             dets_low, scores_low, classes_low,
                             embeds_low if embeds_low is not None else [None] * len(dets_low)
                         )]
        
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
        self.removed_stracks = [t for t in self.removed_stracks if t.state == TrackState.Removed]
        
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
        """Associate tracks with detections."""
        if len(tracks) == 0 or len(detections) == 0:
            return np.empty((0, 2), dtype=int), list(range(len(tracks))), list(range(len(detections)))
        
        # Compute cost matrix
        if self.with_reid and self.method == 1:
            # Use ReID features
            cost_matrix = embedding_distance(tracks, detections, metric='cosine')
            cost_matrix = fuse_motion(self.kalman_filter, cost_matrix, tracks, detections)
        else:
            # Use IoU only
            cost_matrix = iou_distance(tracks, detections)
        
        # Apply threshold
        cost_matrix[cost_matrix > thresh] = thresh + 1e-5
        
        # Solve assignment
        matches, unmatched_tracks, unmatched_dets = linear_assignment(cost_matrix, thresh)
        
        return matches, unmatched_tracks, unmatched_dets
