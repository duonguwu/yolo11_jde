# YOLO-JDE Tracker Documentation

## Tổng quan

YOLO-JDE Tracker là một module độc lập cho Multi-Object Tracking (MOT) sử dụng Joint Detection and Embedding (JDE) approach. Module này được tách ra từ YOLO11-JDE project để tạo thành một package có thể tái sử dụng.

## Kiến trúc hệ thống

### 1. Core Components

```
yolo_jde_tracker/
├── yolo_jde/                          # Main package
│   ├── core/                          # Core inference engine
│   │   ├── inference.py               # JDETracker main class
│   │   ├── postprocess.py             # Post-processing utilities
│   │   └── visualization.py           # Visualization utilities
│   ├── models/                        # JDE model components
│   │   ├── jde_head.py                # JDE Head (Joint Detection & Embedding)
│   │   ├── jde_model.py               # JDE Model wrapper
│   │   ├── predictor.py               # JDE Predictor
│   │   └── yolo_jde.py                # Custom YOLO class with JDE support
│   ├── trackers/                      # Tracking algorithms
│   │   ├── base/basetrack.py          # Base tracking class
│   │   ├── algorithms/                # Tracking algorithms
│   │   │   ├── smile_track.py         # SMILEtrack algorithm
│   │   │   ├── byte_tracker.py        # ByteTracker algorithm
│   │   │   ├── bot_sort.py            # BoTSORT algorithm
│   │   │   └── boost_track.py         # BoostTrack algorithm
│   │   ├── utils/                     # Tracking utilities
│   │   │   ├── kalman_filter.py       # Kalman Filter implementation
│   │   │   ├── matching.py            # Distance calculation & assignment
│   │   │   └── gmc.py                 # Global Motion Compensation
│   │   └── track.py                   # Tracker registration system
│   └── utils/                         # General utilities
│       ├── ops.py                     # Operations (NMS, bbox utils)
│       ├── loss.py                    # JDE Loss functions
│       ├── config.py                  # Configuration management
│       └── torch_utils.py             # PyTorch utilities
├── configs/                           # Configuration files
│   ├── models/                        # Model configurations
│   │   └── yolo11-jde.yaml           # JDE model architecture
│   └── trackers/                      # Tracker configurations
│       ├── smiletrack.yaml           # SMILEtrack config
│       ├── bytetrack.yaml            # ByteTracker config
│       └── botsort.yaml              # BoTSORT config
└── examples/                          # Usage examples
    ├── test_tracking.py              # Video tracking example
    ├── test_detection_coco.py        # COCO detection example
    └── README.md                     # Examples documentation
```

### 2. JDE (Joint Detection and Embedding) Architecture

#### JDE Head
JDE Head là phần mở rộng của YOLO detection head, bao gồm 3 nhánh:

```python
class JDE(Detect):
    def __init__(self, nc=80, embed_dim=128, ch=()):
        super().__init__(nc, ch)
        self.embed_dim = embed_dim
        
        # 3 branches:
        # self.cv2 = Box regression branch (từ Detect)
        # self.cv3 = Classification branch (từ Detect)  
        # self.cv4 = ReID embedding branch (JDE specific)
```

**Output format**: `[batch, anchors, 4+nc+embed_dim]`
- 4: bounding box coordinates (xyxy)
- nc: class probabilities (80 classes)
- embed_dim: ReID embeddings (128-dim vectors)

#### Tracking Pipeline

```
Input Frame → JDE Model → [Boxes, Classes, Embeddings] → Tracker → Tracked Objects
```

1. **Detection**: JDE model tạo ra detections + embeddings
2. **Prediction**: Kalman Filter dự đoán vị trí tracks cũ
3. **Matching**: Tính distance matrix (ReID + Motion + IoU)
4. **Assignment**: Hungarian algorithm để match detections với tracks
5. **Update**: Cập nhật tracks và tạo tracks mới

### 3. Tracking Algorithms

#### SMILEtrack (Recommended)
- **Đặc điểm**: Kết hợp ReID + Motion + IoU
- **Ưu điểm**: Robust với occlusion, accurate với crowded scenes
- **Sử dụng**: Tốt nhất cho JDE models với embeddings

#### ByteTracker
- **Đặc điểm**: Chủ yếu dựa vào IoU + Kalman Filter
- **Ưu điểm**: Nhanh, đơn giản
- **Hạn chế**: Không sử dụng ReID features

#### BoTSORT
- **Đặc điểm**: ByteTracker + ReID + GMC
- **Ưu điểm**: Tốt với camera movement
- **Sử dụng**: Khi có camera motion

## Installation

### Requirements
```bash
# Core dependencies
ultralytics>=8.0.0
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
numpy>=1.21.0
scipy>=1.7.0
pyyaml>=5.4.0
tqdm>=4.62.0
lapx>=0.5.2
pillow>=8.0.0
```

### Install từ source
```bash
git clone <repository-url>
cd yolo_jde_tracker
pip install -e .
```

## Usage

### 1. Basic Usage

#### JDETracker (Recommended)
```python
from yolo_jde import JDETracker

# Initialize tracker
tracker = JDETracker(
    model_path="yolo11s-jde.pt",
    tracker_config="smiletrack.yaml",
    mode="track"  # hoặc "detect"
)

# Track video
results = tracker.track_video(
    source="video.mp4",
    output_path="tracked_video.mp4",
    save_json=True
)

# Track single frame
frame_result = tracker.track_frame(frame)
```

#### YOLO-JDE Class
```python
from yolo_jde import YOLOJDE

# Load model
model = YOLOJDE("yolo11s-jde.pt", task="jde")

# Run tracking
results = model.track(
    source="video.mp4",
    tracker="smiletrack.yaml",
    save=True
)

# Run detection only
results = model.predict("image.jpg")
```

### 2. Detection Only (COCO Format)

```python
from yolo_jde import JDETracker

# Initialize in detect mode
tracker = JDETracker(
    model_path="yolo11s-jde.pt",
    mode="detect"
)

# Detect objects in COCO format
coco_results = tracker.detect_coco_format(
    source="images/",  # Directory or single image
    image_ids=[1, 2, 3]  # Optional image IDs
)

# Save results
import json
with open("detections.json", "w") as f:
    json.dump(coco_results, f, indent=2)
```

### 3. Custom Tracker Configuration

```yaml
# configs/trackers/custom_tracker.yaml
tracker_type: smiletrack
track_high_thresh: 0.6
track_low_thresh: 0.1
new_track_thresh: 0.7
track_buffer: 30
match_thresh: 0.8
proximity_thresh: 0.5
appearance_thresh: 0.25
with_reid: true
```

```python
# Sử dụng custom config
tracker = JDETracker(
    model_path="model.pt",
    tracker_config="custom_tracker.yaml"
)
```

### 4. Advanced Usage

#### Batch Processing
```python
import os
from pathlib import Path

tracker = JDETracker("model.pt", mode="track")

# Process multiple videos
video_dir = Path("videos/")
for video_file in video_dir.glob("*.mp4"):
    output_file = f"tracked_{video_file.name}"
    
    results = tracker.track_video(
        source=str(video_file),
        output_path=output_file,
        save_json=True
    )
    print(f"Processed: {video_file} -> {output_file}")
```

#### Real-time Tracking
```python
import cv2

tracker = JDETracker("model.pt", mode="track")

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Track frame
    result = tracker.track_frame(frame)
    
    # Visualize
    annotated_frame = tracker.visualizer.draw_tracks(
        frame, result['boxes'], result['track_ids'], result['scores']
    )
    
    cv2.imshow("Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Configuration

### Model Configuration
```yaml
# configs/models/jde_model.yaml
conf_thres: 0.25      # Confidence threshold
iou_thres: 0.45       # IoU threshold for NMS
max_det: 1000         # Maximum detections
nc: 80                # Number of classes
embed_dim: 128        # Embedding dimension
imgsz: 640            # Input image size
```

### Tracker Configuration

#### SMILEtrack
```yaml
tracker_type: smiletrack
track_high_thresh: 0.6    # High confidence threshold
track_low_thresh: 0.1     # Low confidence threshold  
new_track_thresh: 0.7     # New track threshold
track_buffer: 30          # Track buffer frames
match_thresh: 0.8         # Matching threshold
proximity_thresh: 0.5     # Proximity threshold
appearance_thresh: 0.25   # Appearance threshold
with_reid: true           # Use ReID features
lambda_: 0.98            # Fusion parameter
```

#### ByteTracker
```yaml
tracker_type: bytetrack
track_thresh: 0.5
track_buffer: 30
match_thresh: 0.8
frame_rate: 30
with_reid: false
```

## API Reference

### JDETracker Class

#### Constructor
```python
JDETracker(
    model_path: Optional[str] = None,
    tracker_config: str = "smiletrack.yaml", 
    device: str = "",
    mode: str = "track",
    **kwargs
)
```

#### Methods

##### track_video()
```python
track_video(
    source: Union[str, int],
    output_path: Optional[str] = None,
    save_json: bool = False,
    json_path: Optional[str] = None,
    show_progress: bool = True
) -> List[Dict[str, Any]]
```

##### track_frame()
```python
track_frame(frame: np.ndarray) -> Dict[str, Any]
```

##### predict_frame()
```python
predict_frame(frame: np.ndarray) -> Dict[str, Any]
```

##### detect_coco_format()
```python
detect_coco_format(
    source: Union[str, List[str]],
    image_ids: Optional[List[int]] = None
) -> List[Dict[str, Any]]
```

### YOLOJDE Class

#### Constructor
```python
YOLOJDE(model="yolo11n.pt", task=None, verbose=False)
```

#### Methods

##### track()
```python
track(
    source=None, 
    stream=False, 
    persist=False, 
    **kwargs
)
```

##### predict()
```python
predict(source=None, **kwargs)
```

## Performance Optimization

### 1. Model Optimization
```python
# Sử dụng TensorRT (nếu có)
model = YOLOJDE("model.pt")
model.export(format="engine")  # TensorRT

# Sử dụng ONNX
model.export(format="onnx")
```

### 2. Inference Optimization
```python
# Batch processing
tracker = JDETracker("model.pt")
tracker.model_config.update({
    'batch': 4,  # Process 4 frames at once
    'half': True,  # Use FP16
})
```

### 3. Memory Optimization
```python
# Giảm track buffer
tracker_config = {
    'track_buffer': 15,  # Giảm từ 30 xuống 15
    'max_det': 500,      # Giảm max detections
}
```

## Troubleshooting

### Common Issues

#### 1. Import Error
```bash
ImportError: No module named 'ultralytics'
```
**Solution**: Install ultralytics
```bash
pip install ultralytics>=8.0.0
```

#### 2. Model Loading Error
```bash
RuntimeError: Failed to load model
```
**Solution**: Check model path và format
```python
# Đảm bảo model path đúng
model_path = "path/to/yolo11s-jde.pt"
assert os.path.exists(model_path), f"Model not found: {model_path}"
```

#### 3. Tracker Initialization Error
```bash
ValueError: Unsupported tracker type
```
**Solution**: Check tracker config
```python
# Supported trackers
supported = ["smiletrack", "bytetrack", "botsort", "boosttrack"]
```

#### 4. CUDA Out of Memory
**Solution**: Reduce batch size hoặc image size
```python
tracker.model_config.update({
    'imgsz': 416,  # Giảm từ 640
    'batch': 1,    # Single frame processing
})
```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

tracker = JDETracker("model.pt", mode="track")
# Sẽ in ra debug information
```

## Examples

Xem thêm examples trong folder `examples/`:
- `test_tracking.py`: Video tracking example
- `test_detection_coco.py`: COCO detection example
- `README.md`: Detailed examples documentation

## Contributing

1. Fork repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

```bibtex
@article{yolo-jde-tracker,
  title={YOLO-JDE Tracker: Joint Detection and Embedding for Multi-Object Tracking},
  author={Your Name},
  year={2024}
}
```
