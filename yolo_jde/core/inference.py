"""Main inference class for YOLO-JDE tracking."""

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple
import json
from tqdm import tqdm

from ..utils.config import ConfigManager
from ..utils.torch_utils import select_device
from .postprocess import PostProcessor
from .visualization import Visualizer
from ..trackers import TRACKER_MAP
from ..trackers.utils import yaml_load
from ultralytics.utils import IterableSimpleNamespace

# Import JDE components
from ..models.yolo_jde import YOLOJDE
from ultralytics import YOLO


class JDETracker:
    """
    Main class for YOLO-JDE tracking and detection.
    
    Supports both tracking mode and detection-only mode.
    """
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 tracker_config: str = "smiletrack.yaml",
                 device: str = "",
                 mode: str = "track",
                 **kwargs):
        """
        Initialize JDE Tracker.
        
        Args:
            model_path: Path to JDE model weights
            tracker_config: Tracker configuration file name
            device: Device to run on ('cpu', 'cuda', 'cuda:0', etc.)
            mode: Mode ('track' for tracking, 'detect' for detection only)
            **kwargs: Additional configuration parameters
        """
        self.mode = mode.lower()
        self.device = select_device(device)
        self.model_path = model_path
        
        # Configuration
        self.config_manager = ConfigManager()
        self.model_config = kwargs.get('model_config', {})
        
        # Initialize model
        self.model = None
        if model_path:
            self.load_model(model_path)
        
        # Initialize post-processor
        self.postprocessor = PostProcessor(
            conf_thres=self.model_config.get('conf_thres', 0.25),
            iou_thres=self.model_config.get('iou_thres', 0.45),
            max_det=self.model_config.get('max_det', 1000),
            nc=self.model_config.get('nc', 80),
            embed_dim=self.model_config.get('embed_dim', 128)
        )
        
        # Initialize visualizer
        self.visualizer = Visualizer()
        
        # Initialize tracker if in tracking mode
        self.tracker = None
        if self.mode == "track":
            self.init_tracker(tracker_config)
    
    def load_model(self, model_path: str):
        """
        Load JDE model using YOLO-JDE class.
        
        Args:
            model_path: Path to model weights
        """
        model_path = Path(model_path)
        
        # Load with custom YOLO-JDE class
        self.model = YOLOJDE(model_path, task="jde")
        self.model.to(self.device)
        print(f"Loaded JDE model: {model_path}")
    
    def init_tracker(self, tracker_config: str):
        """
        Initialize tracker with configuration.
        
        Args:
            tracker_config: Tracker configuration file name
        """
        try:
            config_path = self.config_manager.get_tracker_config_path(tracker_config.replace('.yaml', ''))
            config = yaml_load(config_path) if config_path.exists() else {}
            tracker_type = config.get('tracker_type', 'smiletrack')
            
            if tracker_type not in TRACKER_MAP:
                supported_trackers = list(TRACKER_MAP.keys())
                raise ValueError(f"Unsupported tracker type: {tracker_type}. Supported: {supported_trackers}")
            
            # Convert config to IterableSimpleNamespace for compatibility
            cfg = IterableSimpleNamespace(**config)
            
            # Initialize tracker
            self.tracker = TRACKER_MAP[tracker_type](args=cfg, frame_rate=30)
            print(f"Initialized {tracker_type} tracker")
            
        except Exception as e:
            print(f"Warning: Failed to initialize tracker: {e}")
            print("Falling back to detection-only mode")
            self.mode = "detect"
            self.tracker = None
    
    def predict_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Run prediction on a single frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary containing prediction results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        orig_shape = frame.shape[:2]  # (H, W)
        
        # Run model inference
        if hasattr(self.model, 'predict'):
            # Ultralytics model
            results = self.model.predict(frame, verbose=False)
            if results and len(results) > 0:
                r = results[0]
                
                # Extract predictions
                if hasattr(r, 'boxes') and r.boxes is not None:
                    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes.xyxy is not None else np.array([]).reshape(0, 4)
                    scores = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else np.array([])
                    classes = r.boxes.cls.cpu().numpy().astype(int) if r.boxes.cls is not None else np.array([])
                else:
                    boxes = np.array([]).reshape(0, 4)
                    scores = np.array([])
                    classes = np.array([])
                
                # Extract embeddings if available (JDE model)
                embeddings = np.array([]).reshape(0, 128)  # Default empty embeddings
                if hasattr(r, 'embeddings') and r.embeddings is not None:
                    embeddings = r.embeddings.cpu().numpy()
                elif len(boxes) > 0:
                    # Create dummy embeddings for compatibility
                    embeddings = np.zeros((len(boxes), 128))
                
                input_shape = (640, 640)  # Default YOLO input size
        else:
            # Custom model inference
            # Preprocess frame
            input_tensor = self._preprocess_frame(frame)
            
            # Run inference
            with torch.no_grad():
                predictions = self.model(input_tensor)
            
            # Post-process
            results = self.postprocessor.process(predictions, orig_shape, input_tensor.shape[2:])
            boxes = results['boxes']
            scores = results['scores']
            classes = results['classes']
            embeddings = results['embeddings']
        
        return {
            'boxes': boxes,
            'scores': scores,
            'classes': classes,
            'embeddings': embeddings,
            'frame_shape': orig_shape
        }
    
    def track_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Run tracking on a single frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary containing tracking results
        """
        # Get detections
        detections = self.predict_frame(frame)
        
        if self.mode == "detect" or self.tracker is None:
            # Return detections without tracking
            return {
                **detections,
                'track_ids': np.array([-1] * len(detections['boxes']))
            }
        
        # Run tracker
        try:
            tracked_results = self.tracker.update(detections, frame)
            return tracked_results
        except Exception as e:
            print(f"Tracker failed: {e}, falling back to detection")
            return {
                **detections,
                'track_ids': np.array([-1] * len(detections['boxes']))
            }
    
    def track_video(self,
                   source: Union[str, int],
                   output_path: Optional[str] = None,
                   save_json: bool = False,
                   json_path: Optional[str] = None,
                   show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Track objects in video.
        
        Args:
            source: Video file path or camera index
            output_path: Output video path (optional)
            save_json: Whether to save results as JSON
            json_path: JSON output path (optional)
            show_progress: Whether to show progress bar
            
        Returns:
            List of frame results
        """
        # Open video
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {source}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        
        # Initialize video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        results = []
        frame_idx = 0
        
        pbar = tqdm(total=total_frames, desc="Processing frames") if show_progress else None
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Track frame
                result = self.track_frame(frame)
                
                # Add frame info
                frame_result = {
                    'frame_idx': frame_idx,
                    'objects': self._format_frame_objects(result)
                }
                results.append(frame_result)
                
                # Visualize and save
                if output_path:
                    if self.mode == "track" and 'track_ids' in result:
                        annotated_frame = self.visualizer.draw_tracks(
                            frame, result['boxes'], result['track_ids'], result.get('scores')
                        )
                    else:
                        annotated_frame = self.visualizer.draw_detections(
                            frame, result['boxes'], result.get('scores'), result.get('classes')
                        )
                    writer.write(annotated_frame)
                
                frame_idx += 1
                if pbar:
                    pbar.update(1)
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if pbar:
                pbar.close()
        
        # Save JSON results
        if save_json:
            json_output = json_path or (str(output_path).replace('.mp4', '_results.json') if output_path else 'results.json')
            with open(json_output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved results to: {json_output}")
        
        if output_path:
            print(f"Saved video to: {output_path}")
        
        return results
    
    def detect_coco_format(self, 
                          source: Union[str, List[str]],
                          image_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Run detection and return results in COCO format.
        
        Args:
            source: Image file path, directory, or list of image paths
            image_ids: List of image IDs for COCO format
            
        Returns:
            List of COCO detection dictionaries
        """
        # Handle different source types
        if isinstance(source, str):
            source_path = Path(source)
            if source_path.is_dir():
                # Directory of images
                image_paths = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
                image_paths.sort()
            elif source_path.is_file():
                # Single image
                image_paths = [source_path]
            else:
                raise ValueError(f"Invalid source path: {source}")
        else:
            # List of paths
            image_paths = [Path(p) for p in source]
        
        # Generate image IDs if not provided
        if image_ids is None:
            image_ids = list(range(len(image_paths)))
        
        coco_results = []
        
        for img_path, img_id in tqdm(zip(image_paths, image_ids), total=len(image_paths), desc="Processing images"):
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Cannot load image {img_path}")
                continue
            
            # Run detection
            result = self.predict_frame(image)
            
            # Convert to COCO format
            coco_dets = self.postprocessor.format_coco_detection(result, img_id)
            coco_results.extend(coco_dets)
        
        return coco_results
    
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess frame for model input.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Preprocessed tensor
        """
        # Resize and pad
        img_size = self.model_config.get('imgsz', 640)
        img = cv2.resize(frame, (img_size, img_size))
        
        # Convert BGR to RGB and normalize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return img_tensor.to(self.device)
    
    def _format_frame_objects(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Format frame objects for JSON output.
        
        Args:
            result: Frame tracking/detection result
            
        Returns:
            List of object dictionaries
        """
        objects = []
        boxes = result['boxes']
        scores = result.get('scores', [None] * len(boxes))
        classes = result.get('classes', [None] * len(boxes))
        track_ids = result.get('track_ids', [-1] * len(boxes))
        embeddings = result.get('embeddings', None)
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1
            
            obj = {
                'id': int(track_ids[i]) if track_ids[i] >= 0 else -1,
                'bbox_xyxy': [float(x1), float(y1), float(x2), float(y2)],
                'bbox_xywh': [float(cx), float(cy), float(w), float(h)],
                'conf': float(scores[i]) if scores[i] is not None else None,
                'class': int(classes[i]) if classes[i] is not None else None
            }
            
            # Add embedding if available
            if embeddings is not None:
                if isinstance(embeddings, np.ndarray):
                    if len(embeddings) > i:
                        obj['embedding'] = embeddings[i]
                elif isinstance(embeddings, (list, tuple)) and len(embeddings) > i:
                    obj['embedding'] = embeddings[i]
            
            objects.append(obj)
        
        return objects
