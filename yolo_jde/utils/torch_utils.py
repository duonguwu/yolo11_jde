"""PyTorch utilities."""

import time
import torch
import platform


def select_device(device='', batch=0, newline=False, verbose=True):
    """
    Selects the appropriate PyTorch device based on the provided arguments.

    The function takes a string specifying the device or a torch.device object and returns a torch.device object
    representing the selected device. The function also validates the number of available devices and raises an
    exception if the requested device(s) are not available.

    Args:
        device (str | torch.device, optional): Device string or torch.device object.
            Options are 'None', 'cpu', or 'cuda', or '0' or '0,1,2,3'. Defaults to an empty string, which auto-selects
            the first available GPU, or CPU if no GPU is available.
        batch (int, optional): Batch size being used in your model. Defaults to 0.
        newline (bool, optional): If True, adds a newline at the end of the log string. Defaults to False.
        verbose (bool, optional): If True, logs the device information. Defaults to True.

    Returns:
        (torch.device): Selected device.

    Raises:
        ValueError: If the specified device is not available or if the batch size is not a multiple of the number of
            selected devices.

    Examples:
        >>> select_device('cuda:0')
        device(type='cuda', index=0)

        >>> select_device('cpu')
        device(type='cpu')
    """
    if isinstance(device, torch.device):
        return device

    s = f"YOLO11-JDE 🚀 Python-{platform.python_version()} torch-{torch.__version__} "
    device = str(device).lower()
    for remove in "cuda:", "none", "(", ")", "[", "]", "'", " ":
        device = device.replace(remove, "")  # to string, 'cuda:0' -> '0' and '(0, 1)' -> '0,1'
    cpu = device == "cpu"
    mps = device in ("mps", "mps:0")  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        if device == "cuda":
            device = "0"
        visible = os.environ.get("CUDA_VISIBLE_DEVICES") or "-1"
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable - must be before assert is_available()

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(",") if device else "0"  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        if len(devices) == 1:
            device = torch.device("cuda", int(devices[0]))
        else:
            device = torch.device("cuda:0")
        n = len(devices)  # device count
        if n > 1:  # multi-GPU
            if batch < 1:
                raise ValueError(
                    "batch>=1 to use DataParallel multi-GPU training, "
                    f"i.e. 'python train.py --batch 16 --device 0,1,2,3'"
                )
            if batch % n != 0:
                raise ValueError(
                    f"batch={batch} must be a multiple of GPU count {n}. Try 'python train.py --batch {batch // n * n}' or 'python train.py --batch {(batch // n + 1) * n}'"
                )
        s += f"CUDA:{device} ({', '.join(devices)})\n"  # device info
    elif mps and getattr(torch, "has_mps", False) and torch.backends.mps.is_available() and TORCH_2_0:
        # Prefer MPS if available
        device = torch.device("mps")
        s += f"MPS ({device})\n"
    else:  # revert to CPU
        device = torch.device("cpu")
        s += f"CPU ({device})\n"

    if verbose:
        print(s)
    return device


def time_sync():
    """PyTorch-accurate time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


import os
TORCH_2_0 = int(torch.__version__.split('.')[0]) >= 2  # torch>=2.0.0
