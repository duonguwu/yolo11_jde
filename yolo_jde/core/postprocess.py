"""Post-processing utilities for JDE model outputs."""

import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from ..utils.ops import non_max_suppression, xywh2xyxy, xyxy2xywh


class PostProcessor:
    """Post-processor for JDE model outputs."""
    
    def __init__(self, 
                 conf_thres: float = 0.25,
                 iou_thres: float = 0.45,
                 max_det: int = 1000,
                 nc: int = 80,
                 embed_dim: int = 128):
        """
        Initialize post-processor.
        
        Args:
            conf_thres: Confidence threshold for NMS
            iou_thres: IoU threshold for NMS
            max_det: Maximum detections per image
            nc: Number of classes
            embed_dim: Embedding dimension
        """
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.nc = nc
        self.embed_dim = embed_dim
    
    def process(self, predictions: torch.Tensor, 
                orig_img_shape: Tuple[int, int],
                input_img_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        Process JDE model predictions.
        
        Args:
            predictions: Model output tensor [batch, anchors, 4+nc+embed_dim]
            orig_img_shape: Original image shape (H, W)
            input_img_shape: Input image shape (H, W)
            
        Returns:
            Dictionary containing processed outputs
        """
        # Apply NMS
        nms_outputs = non_max_suppression(
            predictions,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            max_det=self.max_det,
            nc=self.nc
        )
        
        results = []
        for i, output in enumerate(nms_outputs):
            if output is None or len(output) == 0:
                results.append({
                    'boxes': np.array([]).reshape(0, 4),
                    'scores': np.array([]),
                    'classes': np.array([]),
                    'embeddings': np.array([]).reshape(0, self.embed_dim)
                })
                continue
            
            # Split outputs
            boxes = output[:, :4]  # xyxy format
            scores = output[:, 4]
            classes = output[:, 5]
            
            # Extract embeddings if available
            if output.shape[1] > 6:
                embeddings = output[:, 6:6+self.embed_dim]
            else:
                embeddings = torch.zeros((len(output), self.embed_dim), device=output.device)
            
            # Scale boxes to original image size
            boxes = self._scale_boxes(boxes, input_img_shape, orig_img_shape)
            
            # Convert to numpy
            results.append({
                'boxes': boxes.cpu().numpy(),
                'scores': scores.cpu().numpy(),
                'classes': classes.cpu().numpy().astype(int),
                'embeddings': embeddings.cpu().numpy()
            })
        
        return results[0] if len(results) == 1 else results
    
    def _scale_boxes(self, boxes: torch.Tensor, 
                    input_shape: Tuple[int, int], 
                    orig_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Scale boxes from input image size to original image size.
        
        Args:
            boxes: Boxes in xyxy format
            input_shape: Input image shape (H, W)
            orig_shape: Original image shape (H, W)
            
        Returns:
            Scaled boxes
        """
        gain = min(input_shape[0] / orig_shape[0], input_shape[1] / orig_shape[1])
        pad = (input_shape[1] - orig_shape[1] * gain) / 2, (input_shape[0] - orig_shape[0] * gain) / 2
        
        boxes[:, [0, 2]] -= pad[0]  # x padding
        boxes[:, [1, 3]] -= pad[1]  # y padding
        boxes[:, :4] /= gain
        
        # Clip boxes to image bounds
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, orig_shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, orig_shape[0])
        
        return boxes
    
    def format_coco_detection(self, results: Dict[str, Any], 
                            image_id: int, 
                            category_mapping: Optional[Dict[int, int]] = None) -> List[Dict]:
        """
        Format detection results to COCO format.
        
        Args:
            results: Detection results from process()
            image_id: Image ID for COCO format
            category_mapping: Mapping from class indices to COCO category IDs
            
        Returns:
            List of COCO detection dictionaries
        """
        coco_results = []
        
        boxes = results['boxes']  # xyxy format
        scores = results['scores']
        classes = results['classes']
        
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            w, h = x2 - x1, y2 - y1
            
            # Skip invalid boxes
            if w <= 0 or h <= 0:
                continue
            
            category_id = int(classes[i])
            if category_mapping:
                category_id = category_mapping.get(category_id, category_id)
            
            coco_result = {
                'image_id': image_id,
                'category_id': category_id,
                'bbox': [float(x1), float(y1), float(w), float(h)],  # COCO format: [x, y, w, h]
                'score': float(scores[i])
            }
            coco_results.append(coco_result)
        
        return coco_results
