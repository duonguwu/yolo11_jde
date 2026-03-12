"""Global Motion Compensation (GMC) for camera motion compensation."""

import cv2
import numpy as np
from typing import Optional, Tuple


class GMC:
    """
    Global Motion Compensation to handle camera motion in tracking.
    
    Supports multiple methods: ORB, SIFT, ECC, Sparse Optical Flow
    """
    
    def __init__(self, method: str = 'orb', downscale: int = 2):
        """
        Initialize GMC with specified method.
        
        Args:
            method: GMC method ('orb', 'sift', 'ecc', 'sparseOptFlow', 'none')
            downscale: Downscale factor for processing speed
        """
        self.method = method.lower()
        self.downscale = max(1, int(downscale))
        
        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None
        
        self.initializedFirstFrame = False
        
        # Initialize feature detector based on method
        if self.method == 'orb':
            self.detector = cv2.ORB_create(1000)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        elif self.method == 'sift':
            self.detector = cv2.SIFT_create(1000)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        elif self.method == 'ecc':
            self.warp_mode = cv2.MOTION_EUCLIDEAN
            self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-6)
        elif self.method == 'sparseoptflow':
            self.feature_params = dict(
                maxCorners=1000,
                qualityLevel=0.01,
                minDistance=1,
                blockSize=3,
                useHarrisDetector=False,
                k=0.04
            )
    
    def apply(self, raw_frame: np.ndarray, detections: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply global motion compensation.
        
        Args:
            raw_frame: Input frame
            detections: Optional detections to transform
            
        Returns:
            Homography matrix (3x3) or identity if no motion detected
        """
        if self.method == 'none':
            return np.eye(3, dtype=np.float32)
        
        # Downscale frame for processing
        height, width = raw_frame.shape[:2]
        frame = cv2.resize(raw_frame, (width // self.downscale, height // self.downscale))
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        H = np.eye(3, dtype=np.float32)
        
        if not self.initializedFirstFrame:
            self.prevFrame = frame.copy()
            self.initializedFirstFrame = True
            return H
        
        try:
            if self.method in ['orb', 'sift']:
                H = self._feature_based_gmc(frame)
            elif self.method == 'ecc':
                H = self._ecc_based_gmc(frame)
            elif self.method == 'sparseoptflow':
                H = self._optical_flow_gmc(frame)
        except Exception as e:
            print(f"GMC failed: {e}")
            H = np.eye(3, dtype=np.float32)
        
        # Scale homography back to original resolution
        if self.downscale > 1:
            H[0, 2] *= self.downscale
            H[1, 2] *= self.downscale
        
        self.prevFrame = frame.copy()
        return H
    
    def _feature_based_gmc(self, frame: np.ndarray) -> np.ndarray:
        """Feature-based GMC using ORB or SIFT."""
        # Detect keypoints and descriptors
        keypoints, descriptors = self.detector.detectAndCompute(frame, None)
        
        if descriptors is None or len(keypoints) < 10:
            return np.eye(3, dtype=np.float32)
        
        # Match with previous frame
        if self.prevDescriptors is not None:
            matches = self.matcher.knnMatch(self.prevDescriptors, descriptors, k=2)
            
            # Filter good matches using Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) >= 10:
                # Extract matched points
                src_pts = np.float32([self.prevKeyPoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Find homography
                H, mask = cv2.findHomography(src_pts, dst_pts, 
                                           cv2.RANSAC, 5.0, maxIters=5000, confidence=0.995)
                
                if H is not None:
                    self.prevKeyPoints = keypoints
                    self.prevDescriptors = descriptors
                    return H.astype(np.float32)
        
        self.prevKeyPoints = keypoints
        self.prevDescriptors = descriptors
        return np.eye(3, dtype=np.float32)
    
    def _ecc_based_gmc(self, frame: np.ndarray) -> np.ndarray:
        """ECC-based GMC using Enhanced Correlation Coefficient."""
        if self.prevFrame is None:
            return np.eye(3, dtype=np.float32)
        
        # Initialize warp matrix
        if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        try:
            # Run ECC algorithm
            (cc, warp_matrix) = cv2.findTransformECC(
                self.prevFrame, frame, warp_matrix, self.warp_mode, self.criteria, None, 1
            )
            
            # Convert to homography matrix
            if self.warp_mode != cv2.MOTION_HOMOGRAPHY:
                H = np.eye(3, dtype=np.float32)
                H[:2, :] = warp_matrix
                return H
            else:
                return warp_matrix.astype(np.float32)
                
        except Exception:
            return np.eye(3, dtype=np.float32)
    
    def _optical_flow_gmc(self, frame: np.ndarray) -> np.ndarray:
        """Sparse optical flow based GMC."""
        if self.prevFrame is None:
            return np.eye(3, dtype=np.float32)
        
        # Detect corners in previous frame
        prev_corners = cv2.goodFeaturesToTrack(self.prevFrame, **self.feature_params)
        
        if prev_corners is None or len(prev_corners) < 10:
            return np.eye(3, dtype=np.float32)
        
        # Calculate optical flow
        next_corners, status, error = cv2.calcOpticalFlowPyrLK(
            self.prevFrame, frame, prev_corners, None,
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Filter good points
        good_prev = prev_corners[status == 1]
        good_next = next_corners[status == 1]
        
        if len(good_prev) >= 10:
            # Find homography
            H, mask = cv2.findHomography(good_prev, good_next, 
                                       cv2.RANSAC, 5.0, maxIters=5000, confidence=0.995)
            
            if H is not None:
                return H.astype(np.float32)
        
        return np.eye(3, dtype=np.float32)
    
    def reset(self):
        """Reset GMC state."""
        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None
        self.initializedFirstFrame = False
