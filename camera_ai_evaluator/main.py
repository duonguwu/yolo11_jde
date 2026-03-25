import os
import json
import yaml
import argparse

from camera_ai_evaluator import annotation_io
from camera_ai_evaluator.detection.metrics_runner import MetricsRunner as DetectionMetricsRunner
from camera_ai_evaluator.entity.dataset import DetectedDataset
from camera_ai_evaluator.annotation_io.coco import CocoDetectionAnnotationIO
from camera_ai_evaluator.person_infomation.annalyze_gt import (
    crop_and_save_selected_attribute,
)


def run_detection(config: dict):
    runner = DetectionMetricsRunner(
        is_visualize=config["eval_params"]["is_visualize"],
        iou_threshold=config["eval_params"]["iou_threshold"],
        score_threshold=config["eval_params"].get("score_threshold", 0.0),
    )
    annotation_io = CocoDetectionAnnotationIO()

    for dataset_id, dataset_cfg in config["datasets"].items():
        print(f"Evaluating dataset: {dataset_id}")
        if dataset_cfg["gt_type"] != "COCO" or dataset_cfg["pd_type"] != "COCO":
            raise NotImplementedError(
                "Currently only COCO format is supported for both GT and PD."
            )
        gt_dataset = annotation_io.load_annotations(dataset_cfg["gt_path"])
        pd_dataset = annotation_io.load_annotations(dataset_cfg["pred_path"])
        gt_dataset.re_id_frames_by_image_name()
        pd_dataset.re_id_frames_by_image_name()

        # load image_name_to_path mapping
        img_name_to_path = dataset_cfg.get("image_name_to_path", {})
        if img_name_to_path:
            with open(img_name_to_path, "r") as f:
                img_name_to_path = json.load(f)

        # clear all contents in output directory
        if os.path.exists(dataset_cfg["output_dir"]):
            os.system(f"rm -rf {dataset_cfg['output_dir']}/*")

        os.makedirs(dataset_cfg["output_dir"], exist_ok=True)

        runner.run(
            gts=gt_dataset,
            preds=pd_dataset,
            output_dir=dataset_cfg["output_dir"],
            img_name_to_path=img_name_to_path,
        )


def run_person_info_checking(config: dict):
    annotation_io = CocoDetectionAnnotationIO()
    # load image_name_to_path mapping
    for dataset_id, dataset_cfg in config["datasets"].items():
        print(f"Evaluating dataset: {dataset_id}")
        if dataset_cfg["type"] != "COCO":
            raise NotImplementedError("Currently only COCO format is supported.")

        dataset = annotation_io.load_annotations(dataset_cfg["label_path"])

        img_name_to_path = dataset_cfg.get("image_name_to_path", {})
        if img_name_to_path:
            with open(img_name_to_path, "r") as f:
                img_name_to_path = json.load(f)

        # clear all contents in output directory
        if os.path.exists(dataset_cfg["output_dir"]):
            os.system(f"rm -rf {dataset_cfg['output_dir']}/*")
        crop_and_save_selected_attribute(
            dataset=dataset,
            image_map=img_name_to_path,
            output_dir=dataset_cfg["output_dir"],
            consider_attrs=config["eval_params"]["consider_attrs"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera AI Evaluator")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=False,
        help="Path to configuration file",
    )
    args = parser.parse_args()
    config_path = (
        args.config if args.config else "config/camera_ai_evaluator/config.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config["type"] == "detection":
        run_detection(config)
    if config["type"] == "person_info_checking":
        run_person_info_checking(config)
