from dataclasses import dataclass


@dataclass(slots=True)
class SequenceInfo:
    """Information about a video or a sequence of frames."""

    name: str
    total_frames: int
    frame_rate: float
    width: int
    height: int
