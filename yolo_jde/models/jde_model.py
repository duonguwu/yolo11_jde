"""JDE Model implementation - Joint Detection and Embedding."""

from ultralytics.nn.tasks import DetectionModel
from ..utils.loss import 11JDELoss


class JDEModel(DetectionModel):
    """YOLO11 Joint Detection and Embedding (JDE) model."""

    def __init__(self, cfg="yolo11n-jde.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLO11 JDE model with given config and parameters."""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the JDEModel."""
        return 11JDELoss(self)
