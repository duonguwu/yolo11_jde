from abc import ABC, abstractmethod

from camera_ai_evaluator.entity.dataset import (
    DetectedDataset,
)


class DetectionAnnotationIO(ABC):
    """
    Abtract class to load and save annotation
    """

    @abstractmethod
    def load_annotations(self, annotation_file_path: str) -> DetectedDataset:
        raise NotImplementedError

    @abstractmethod
    def save_annotations(self, annotations: DetectedDataset) -> None:
        raise NotImplementedError
