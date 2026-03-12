# Examples - YOLO-JDE Tracker

Các ví dụ sử dụng module YOLO-JDE Tracker.

## 1. Video Tracking (test_tracking.py)

Tracking đối tượng trong video, tương tự như `track_save.py` gốc.

### Sử dụng cơ bản:
```bash
python test_tracking.py --model path/to/model.pt --source path/to/video.mp4
```

### Tùy chọn đầy đủ:
```bash
python test_tracking.py \
    --model models/YOLO11s_JDE-CHMOT17.pt \
    --source ../videos/test_video.mp4 \
    --output outputs/tracked_video.mp4 \
    --tracker smiletrack.yaml \
    --device cuda:0 \
    --conf-thres 0.25 \
    --iou-thres 0.45 \
    --save-json \
    --json-path outputs/results.json
```

### Tham số:
- `--model`: Đường dẫn đến model JDE
- `--source`: Video đầu vào
- `--output`: Video đầu ra (tự động tạo nếu không chỉ định)
- `--tracker`: Cấu hình tracker (smiletrack.yaml, bytetrack.yaml)
- `--device`: Device chạy (cpu, cuda, cuda:0, ...)
- `--conf-thres`: Ngưỡng confidence
- `--iou-thres`: Ngưỡng IoU cho NMS
- `--save-json`: Lưu kết quả JSON
- `--json-path`: Đường dẫn file JSON

## 2. Detection COCO Format (test_detection_coco.py)

Chạy detection và xuất kết quả theo format COCO để đánh giá.

### Sử dụng cơ bản:
```bash
python test_detection_coco.py --model path/to/model.pt --source path/to/images/
```

### Tùy chọn đầy đủ:
```bash
python test_detection_coco.py \
    --model models/YOLO11s_JDE-CHMOT17.pt \
    --source ../images/ \
    --output detections.json \
    --device cuda:0 \
    --conf-thres 0.25 \
    --iou-thres 0.45 \
    --image-ids image_ids.json
```

### Tham số:
- `--model`: Đường dẫn đến model JDE
- `--source`: Ảnh đầu vào (file, thư mục, hoặc danh sách)
- `--output`: File JSON đầu ra (tự động tạo nếu không chỉ định)
- `--device`: Device chạy
- `--conf-thres`: Ngưỡng confidence
- `--iou-thres`: Ngưỡng IoU cho NMS
- `--image-ids`: File JSON chứa mapping image IDs

### Format COCO Output:
```json
{
  "images": [
    {
      "id": 0,
      "file_name": "image1.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "image_id": 0,
      "category_id": 0,
      "bbox": [x, y, width, height],
      "score": 0.95
    }
  ],
  "categories": [
    {
      "id": 0,
      "name": "person",
      "supercategory": "object"
    }
  ]
}
```

## 3. Sử dụng trong Python

### Video Tracking:
```python
from yolo_jde import JDETracker

# Khởi tạo tracker
tracker = JDETracker(
    model_path="models/yolo11s_jde.pt",
    tracker_config="smiletrack.yaml",
    mode="track"
)

# Track video
results = tracker.track_video(
    source="video.mp4",
    output_path="tracked_video.mp4",
    save_json=True
)
```

### Detection Only:
```python
from yolo_jde import JDETracker

# Khởi tạo detector
detector = JDETracker(
    model_path="models/yolo11s_jde.pt",
    mode="detect"
)

# Detect và xuất COCO format
coco_results = detector.detect_coco_format("images/")
```

### Single Frame Processing:
```python
import cv2
from yolo_jde import JDETracker

tracker = JDETracker(model_path="model.pt", mode="track")

# Đọc frame
frame = cv2.imread("image.jpg")

# Track frame
result = tracker.track_frame(frame)

print(f"Found {len(result['boxes'])} objects")
print(f"Track IDs: {result['track_ids']}")
```

## 4. Cấu hình Tracker

### SmileTrack (configs/trackers/smiletrack.yaml):
```yaml
tracker_type: smiletrack
track_high_thresh: 0.6
track_low_thresh: 0.1
new_track_thresh: 0.7
track_buffer: 30
match_thresh: 0.8
with_reid: True
proximity_thresh: 0.5
appearance_thresh: 0.25
method: 1
```

### ByteTracker (configs/trackers/bytetrack.yaml):
```yaml
tracker_type: bytetrack
track_high_thresh: 0.6
track_low_thresh: 0.1
new_track_thresh: 0.7
track_buffer: 30
match_thresh: 0.8
with_reid: False
```

## 5. Lỗi thường gặp

### Model không load được:
- Kiểm tra đường dẫn model
- Đảm bảo model là JDE format
- Thử với device khác (cpu thay vì cuda)

### Tracker không hoạt động:
- Kiểm tra file config tracker
- Thử với tracker khác (bytetrack thay vì smiletrack)
- Chạy ở mode "detect" để test model trước

### Performance chậm:
- Sử dụng GPU (--device cuda)
- Giảm resolution input
- Tăng confidence threshold
- Sử dụng ByteTracker thay vì SmileTrack
