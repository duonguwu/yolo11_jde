"""Custom YOLO class with JDE task support."""

from ultralytics import YOLO
# from ultralytics.engine.model import Model

from .jde_model import JDEModel
from .predictor import JDEPredictor
from ..trackers.track import register_tracker


class YOLOJDE(YOLO):
    """
    YOLO-JDE model class that extends YOLO to support Joint Detection and Embedding tasks.
    
    Example:
        ```python
        from yolo_jde import YOLOJDE
        
        # Load JDE model
        model = YOLOJDE("yolo11n-jde.pt", task="jde")
        
        # Run inference
        results = model.predict("path/to/video.mp4")
        
        # Run tracking
        results = model.track("path/to/video.mp4", tracker="smiletrack.yaml")
        ```
    """
    
    def __init__(self, model="yolo11n.pt", task=None, verbose=False):
        """Initialize YOLO-JDE model."""
        # Force task to jde if not specified
        if task is None:
            task = "jde"
        
        super().__init__(model=model, task=task, verbose=verbose)
    
    @property
    def task_map(self):
        """Add JDE task to task mapping."""
        task_map = super().task_map
        task_map.update({
            "jde": {
                "model": JDEModel,
                "predictor": JDEPredictor,
                # trainer và validator không cần cho inference
            }
        })
        return task_map
    
    def track(self, source=None, stream=False, persist=False, **kwargs):
        """
        Perform object tracking on the given input source using the registered trackers.

        Args:
            source (str, optional): The input source for object tracking. Can be a file path, URL, or other
                supported formats.
            stream (bool, optional): If True, enables real-time streaming. Defaults to False.
            persist (bool, optional): If True, persists the trackers between different calls. Defaults to False.
            **kwargs (optional): Additional keyword arguments for the tracking process.

        Returns:
            (List[ultralytics.engine.results.Results]): The tracking results.

        Examples:
            Track objects in a video file:
            >>> model = YOLOJDE('yolo11n-jde.pt')
            >>> results = model.track('path/to/video.mp4')
            >>> for r in results:
            ...     print(r.boxes.id)  # print tracking IDs
        """
        if not hasattr(self.predictor, 'trackers'):
            register_tracker(self, persist)
        kwargs['mode'] = 'track'
        return self.predict(source=source, stream=stream, **kwargs)
