"""Visualization utilities for tracking results."""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict


class Visualizer:
    """Visualizer for tracking and detection results."""
    
    def __init__(self,
                 bbox_color: Tuple[int, int, int] = (0, 255, 0),
                 bbox_thickness: int = 2,
                 text_color: Tuple[int, int, int] = (255, 255, 255),
                 text_bg_color: Tuple[int, int, int] = (0, 0, 0),
                 text_scale: float = 0.4,
                 text_thickness: int = 1,
                 text_padding: int = 2,
                 line_color: Tuple[int, int, int] = (230, 230, 230),
                 line_thickness: int = 4):
        """
        Initialize visualizer.
        
        Args:
            bbox_color: Bounding box color (B, G, R)
            bbox_thickness: Bounding box line thickness
            text_color: Text color (B, G, R)
            text_bg_color: Text background color (B, G, R)
            text_scale: Text scale factor
            text_thickness: Text line thickness
            text_padding: Text padding in pixels
            line_color: Track line color (B, G, R)
            line_thickness: Track line thickness
        """
        self.bbox_color = bbox_color
        self.bbox_thickness = bbox_thickness
        self.text_color = text_color
        self.text_bg_color = text_bg_color
        self.text_scale = text_scale
        self.text_thickness = text_thickness
        self.text_padding = text_padding
        self.line_color = line_color
        self.line_thickness = line_thickness
        
        # Track history for drawing trails
        self.track_history = defaultdict(list)
        self.max_track_history = 30
    
    def draw_detections(self, 
                       image: np.ndarray,
                       boxes: np.ndarray,
                       scores: Optional[np.ndarray] = None,
                       classes: Optional[np.ndarray] = None,
                       class_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Draw detection results on image.
        
        Args:
            image: Input image
            boxes: Bounding boxes in xyxy format
            scores: Confidence scores
            classes: Class indices
            class_names: List of class names
            
        Returns:
            Annotated image
        """
        annotated_img = image.copy()
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            
            # Draw bounding box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), self.bbox_color, self.bbox_thickness)
            
            # Prepare label text
            label_parts = []
            if classes is not None and class_names is not None:
                class_name = class_names[int(classes[i])] if int(classes[i]) < len(class_names) else f"cls_{int(classes[i])}"
                label_parts.append(class_name)
            
            if scores is not None:
                label_parts.append(f"{scores[i]:.2f}")
            
            if label_parts:
                label_text = " ".join(label_parts)
                self._draw_text_with_background(annotated_img, label_text, (x1, y1))
        
        return annotated_img
    
    def draw_tracks(self,
                   image: np.ndarray,
                   boxes: np.ndarray,
                   track_ids: np.ndarray,
                   scores: Optional[np.ndarray] = None,
                   draw_trails: bool = True) -> np.ndarray:
        """
        Draw tracking results on image.
        
        Args:
            image: Input image
            boxes: Bounding boxes in xyxy format
            track_ids: Track IDs
            scores: Confidence scores
            draw_trails: Whether to draw track trails
            
        Returns:
            Annotated image
        """
        annotated_img = image.copy()
        
        for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
            x1, y1, x2, y2 = map(int, box)
            
            # Draw bounding box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), self.bbox_color, self.bbox_thickness)
            
            # Draw track ID
            if track_id >= 0:
                label_text = str(int(track_id))
                self._draw_text_with_background(annotated_img, label_text, (x1, y1))
                
                # Update track history
                if draw_trails:
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    track = self.track_history[track_id]
                    track.append((center_x, center_y))
                    if len(track) > self.max_track_history:
                        track.pop(0)
                    
                    # Draw track trail
                    if len(track) >= 2:
                        points = np.array(track, dtype=np.int32).reshape(-1, 1, 2)
                        cv2.polylines(annotated_img, [points], isClosed=False, 
                                    color=self.line_color, thickness=self.line_thickness)
        
        return annotated_img
    
    def _draw_text_with_background(self, 
                                  image: np.ndarray, 
                                  text: str, 
                                  position: Tuple[int, int]) -> None:
        """
        Draw text with background on image.
        
        Args:
            image: Image to draw on
            text: Text to draw
            position: Text position (x, y)
        """
        x, y = position
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
        )
        
        # Calculate text position (top-left corner of bbox, inside if not enough space above)
        total_text_height = text_height + self.text_padding * 2
        if y >= total_text_height:
            # Place text above bbox
            text_y = y - self.text_padding
            bg_y1 = y - total_text_height
        else:
            # Place text inside bbox at top
            text_y = y + text_height + self.text_padding
            bg_y1 = y + self.text_padding
        
        # Draw background rectangle
        bg_x1 = x - self.text_padding
        bg_x2 = x + text_width + self.text_padding
        bg_y2 = bg_y1 + total_text_height
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), self.text_bg_color, -1)
        
        # Draw text
        cv2.putText(
            image, text, (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_color,
            self.text_thickness, cv2.LINE_AA
        )
    
    def reset_track_history(self):
        """Reset track history."""
        self.track_history.clear()
    
    def cleanup_track_history(self, active_track_ids: List[int]):
        """
        Remove inactive tracks from history.
        
        Args:
            active_track_ids: List of currently active track IDs
        """
        inactive_ids = [tid for tid in self.track_history.keys() if tid not in active_track_ids]
        for tid in inactive_ids:
            del self.track_history[tid]
