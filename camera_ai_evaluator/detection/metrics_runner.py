import os
import cv2
import pandas as pd
from collections import defaultdict

from camera_ai_evaluator.entity.frame import FrameId
from camera_ai_evaluator.entity.dataset import DetectedDataset
from camera_ai_evaluator.detection.metrics import (
    MatchResult,
    analyze_matches_detailed,
    cal_mAP,
    cal_f1,
    cal_precision,
    cal_recall,
    cal_mAP_IoU_threshold,
    calculate_tp_fp_fn,
)
from camera_ai_evaluator.helper.logger import get_logger
from camera_ai_evaluator.helper.drawing import draw_box

logger = get_logger(__name__)


class MetricsRunner:
    def __init__(
        self, 
        is_visualize: bool = True, 
        iou_threshold: float = 0.5,
        score_threshold: float = 0.0
    ) -> None:
        self.is_visualize = is_visualize
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

    def run(
        self,
        gts: DetectedDataset,
        preds: DetectedDataset,
        output_dir: str = "evaluation_results",
        img_name_to_path: dict[str, str] | None = None,
    ):
        """Run entire evaluation."""
        logger.info("1. Verifying input annotations...")
        self._verified_input(gts, preds)
        
        # Filter predictions by score threshold if specified
        if self.score_threshold > 0.0:
            logger.info(f"Filtering predictions with score >= {self.score_threshold}...")
            original_pred_count = sum(len(frame.detected_objects) for frame in preds.detected_frames.values())
            preds = preds.filter_by_score(self.score_threshold)
            filtered_pred_count = sum(len(frame.detected_objects) for frame in preds.detected_frames.values())
            logger.info(f"Filtered predictions: {original_pred_count} -> {filtered_pred_count} (removed {original_pred_count - filtered_pred_count} low-confidence predictions)")

        logger.info("1. Creating output directories...")
        result_dir, fp_dir, fn_dir = self._create_output_dirs(output_dir)

        logger.info("2. Performing detailed match analysis...")
        match_results = analyze_matches_detailed(
            gts=gts, preds=preds, iou_threshold=self.iou_threshold
        )

        logger.info("3. Calculating and saving metrics...")
        self.calculate_and_save_metrics(
            gts=gts,
            preds=preds,
            output_dir=output_dir,
        )

        if self.is_visualize:
            if img_name_to_path is None:
                raise ValueError(
                    "image name to path must be provided for visualization."
                )

            logger.info(f"4. Visualizing errors (FP/FN) at IoU={self.iou_threshold}...")
            self.visualize_errors(
                gts=gts,
                match_results=match_results,
                img_name_to_path=img_name_to_path,
                fp_dir=fp_dir,
                fn_dir=fn_dir,
            )

            logger.info(f"5. Visualizing overall results...")
            self.visualize_overall_results(
                gts=gts,
                preds=preds,
                img_name_to_path=img_name_to_path,
                results_dir=result_dir,
            )

            logger.info(f"\nEvaluation finished. Results are saved in '{output_dir}'")

    def _verified_input(self, gts: DetectedDataset, preds: DetectedDataset) -> bool:
        """Verify input annotations."""
        if self.is_visualize:
            for gt in gts.detected_frames.values():
                if gt.image_name is None and gt.video_name is None:
                    logger.error(
                        f"We need image_name or video_name in GT frame id={gt.id} for visualization."
                    )
                    return False
            for pred in preds.detected_frames.values():
                if pred.image_name is None and pred.video_name is None:
                    logger.error(
                        f"We need image_name or video_name in Pred frame id={pred.id} for visualization."
                    )
                    return False
        return True

    def _create_output_dirs(self, output_dir: str) -> tuple[str, str, str]:
        """
        Create necessary output directories.

        Args:
            output_dir (str): main output directory

        Returns:
            tuple[str, str, str]: results_dir, fp_dir, fn_dir
        """
        results_dir = os.path.join(output_dir, "overall_visuals")
        errors_dir = os.path.join(output_dir, "errors")
        fp_dir = os.path.join(errors_dir, "FP")
        fn_dir = os.path.join(errors_dir, "FN")

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(errors_dir, exist_ok=True)
        os.makedirs(fp_dir, exist_ok=True)
        os.makedirs(fn_dir, exist_ok=True)

        return results_dir, fp_dir, fn_dir

    def calculate_and_save_metrics(
        self,
        gts: DetectedDataset,
        preds: DetectedDataset,
        output_dir: str,
    ):
        precision = cal_precision(gts, preds, self.iou_threshold)
        recall = cal_recall(gts, preds, self.iou_threshold)
        f1 = cal_f1(gts, preds, self.iou_threshold)

        map_50 = cal_mAP_IoU_threshold(gts, preds, 0.5)
        map_coco = cal_mAP(gts, preds)

        # Tạo DataFrame
        summary_data = {
            "Metric": [
                f"Precision @{ self.iou_threshold}",
                f"Recall @{ self.iou_threshold}",
                f"F1-Score @{ self.iou_threshold}",
                "mAP @.50",
                "mAP @.50-.95 (COCO)",
            ],
            "Value": [precision, recall, f1, map_50, map_coco],
        }
        summary_df = pd.DataFrame(summary_data)

        # Chi tiết TP/FP/FN theo class
        stats_by_class = calculate_tp_fp_fn(gts, preds, self.iou_threshold)
        class_df = pd.DataFrame.from_dict(stats_by_class, orient="index")
        class_df = class_df.reset_index().rename(columns={"index": "ClassID"})

        # Lưu vào CSV
        summary_df.to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False)
        class_df.to_csv(os.path.join(output_dir, "class_metrics.csv"), index=False)

    def visualize_errors(
        self,
        gts: DetectedDataset,
        match_results: MatchResult,
        img_name_to_path: dict[str, str],
        fp_dir: str,
        fn_dir: str,
    ):
        # Visualize False Positives
        for i, fp_obj in enumerate(match_results.false_positives):
            image_name = str(gts.detected_frames[fp_obj.frame_id].image_name)
            image_path = img_name_to_path.get(image_name, "")
            if not os.path.exists(image_path):
                print(f"Warning: Image not found for FP: {image_path}")
                continue

            image = cv2.imread(image_path)
            label = f"FP: {fp_obj.class_id}"
            # Màu đỏ cho FP
            draw_box(
                image, fp_obj.bounding_box.coord, label, (0, 0, 255), box_type="xywh"
            )

            output_path = os.path.join(fp_dir, f"fp_{image_name}_{i}.jpg")
            cv2.imwrite(output_path, image)

        # Visualize False Negatives
        for i, fn_obj in enumerate(match_results.false_negatives):
            image_name = str(gts.detected_frames[fn_obj.frame_id].image_name)
            image_path = img_name_to_path.get(image_name, "")
            if not os.path.exists(image_path):
                print(f"Warning: Image not found for FN: {image_path}")
                continue

            image = cv2.imread(image_path)
            label = f"FN: {fn_obj.class_id}"
            # Màu vàng cho FN
            draw_box(
                image, fn_obj.bounding_box.coord, label, (0, 255, 255), box_type="xywh"
            )

            output_path = os.path.join(fn_dir, f"fp_{image_name}_{i}.jpg")
            cv2.imwrite(output_path, image)

    def visualize_overall_results(
        self,
        gts: DetectedDataset,
        preds: DetectedDataset,
        img_name_to_path: dict[str, str],
        results_dir: str,
    ):
        """Vẽ tất cả GT và Pred lên ảnh để so sánh tổng quan."""
        # Gộp tất cả các object theo frame_id
        all_objects_by_frame = defaultdict(lambda: {"gts": [], "preds": []})
        for fa in gts.detected_frames.values():
            all_objects_by_frame[fa.id]["gts"].extend(fa.detected_objects)
        for fa in preds.detected_frames.values():
            all_objects_by_frame[fa.id]["preds"].extend(fa.detected_objects)

        for frame_id, data in all_objects_by_frame.items():
            image_name = str(gts.detected_frames[frame_id].image_name)
            image_path = img_name_to_path.get(image_name, "")
            if not os.path.exists(image_path):
                continue

            image = cv2.imread(image_path)

            # Vẽ GT boxes (màu xanh lá)
            for gt_obj in data["gts"]:
                label = f"GT: {gt_obj.class_id}"
                draw_box(
                    image,
                    gt_obj.bounding_box.coord,
                    label,
                    (0, 255, 0),
                    box_type="xywh",
                    label_position="bottom_right",
                )

            # Vẽ Pred boxes (màu xanh dương)
            for pred_obj in data["preds"]:
                label = f"Pred: {pred_obj.class_id}"
                draw_box(
                    image,
                    pred_obj.bounding_box.coord,
                    label,
                    (255, 0, 0),
                    box_type="xywh",
                    label_position="top_left",
                )

            # NOTE 14/11/2025: Vẽ nhãn nếu là nhân viên, bảo vệ để phục vụ kiểm tra chất lượng
            # gán nhãn
            for gt_obj in data["gts"]:
                if gt_obj.attributes["is_staff"]:
                    label = f"STAFF"
                    draw_box(
                        image,
                        gt_obj.bounding_box.coord,
                        label,
                        (0, 255, 255),
                        box_type="xywh",
                        label_position="center",
                    )
                elif gt_obj.attributes["is_security"]:
                    label = f"SECURITY"
                    draw_box(
                        image,
                        gt_obj.bounding_box.coord,
                        label,
                        (255, 255, 0),
                        box_type="xywh",
                        label_position="center",
                    )

            output_path = os.path.join(results_dir, f"{image_name}.jpg")
            cv2.imwrite(output_path, image)
