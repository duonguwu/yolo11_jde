import os
from PIL import Image
from tqdm import tqdm
from pathlib import Path

from camera_ai_evaluator.annotation_io.coco import CocoDetectionAnnotationIO
from camera_ai_evaluator.entity.dataset import DetectedDataset
from camera_ai_evaluator.entity.object import AttributeName


def crop_and_save_selected_attribute(
    dataset: DetectedDataset,
    image_map: dict[str, str],
    output_dir: str,
    consider_attrs: list[AttributeName],
) -> None:
    for fa in tqdm(dataset.detected_frames.values()):
        image_path = image_map[str(fa.image_name)]
        image = Image.open(image_path)

        for obj in fa.detected_objects:
            bounding_box = obj.bounding_box
            if bounding_box.coord_type == "xywh":
                bounding_box.change_coord_type("xyxy")

            crop_box = bounding_box.coord

            cropped_image = image.crop(crop_box.tolist())

            for attr_name in consider_attrs:
                if attr_name in obj.attributes:
                    attr_value_str = obj.attributes[attr_name]
                    attr_output_dir = Path(output_dir) / attr_name / attr_value_str
                    attr_output_dir.mkdir(parents=True, exist_ok=True)

                    # Ví dụ: original_image_name_ann154.jpg
                    base_filename, _ = os.path.splitext(fa.image_name)
                    output_filename = f"{base_filename}_obj_{obj.id}.jpg"
                    output_path = attr_output_dir / output_filename

                    # Lưu ảnh đã cắt
                    cropped_image.save(output_path, "JPEG")
