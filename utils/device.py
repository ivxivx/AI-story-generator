from enum import Enum
from typing import Optional, Union
import torch

class DeviceType(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


def get_device_type() -> DeviceType:
    device: DeviceType

    if torch.cuda.is_available():
        device = DeviceType.CUDA

    elif torch.backends.mps.is_available():
        device = DeviceType.MPS

    else:
        device = DeviceType.CPU
    
    print(f"Using device: {device}")
    return device


def get_torch_dtype(device: DeviceType) -> Optional[torch.dtype]:
    dtype = torch.float16 if device == DeviceType.CUDA or device == DeviceType.MPS else None
    
    print(f"Using data type: {dtype}")
    return dtype
