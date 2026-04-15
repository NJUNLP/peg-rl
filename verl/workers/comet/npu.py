import os
import logging

import torch
import torch_npu
from functools import lru_cache
from typing import Any, Dict, Union
from pytorch_lightning.accelerators.accelerator import Accelerator

from typing_extensions import override

_log = logging.getLogger(__name__)

class NPUAccelerator(Accelerator):
    """Accelerator for HUAWEI NPU devices."""

    @override
    def setup_device(self, device: torch.device) -> None:
        """
        Raises:
            ValueError: If the selected device is not of type NPU.
        """
        if device.type != "npu":
            raise ValueError(
                f"Device should be of type 'npu', got '{device.type}' instead."
            )
        if device.index is None:
            device = torch.device("npu", 0)
        torch.npu.set_device(device.index)

    @override
    def teardown(self) -> None:
        torch.npu.empty_cache()

    @staticmethod
    @override
    def parse_devices(devices: Any) -> Any:
        return [torch.device("npu", i) for i in range(torch.npu.device_count())]

    @staticmethod
    @override
    def get_parallel_devices(devices: Any) -> Any:
        if isinstance(devices, int):
            return [torch.device("npu", i) for i in range(devices)]
        elif isinstance(devices, list):
            try:
                return [torch.device("npu", i) for i in devices]
            except Exception:
                return devices
        elif devices in ("auto", "npu"):
            return [torch.device("npu", i) for i in range(torch.npu.device_count())]
        return []

    @staticmethod
    @override
    def auto_device_count() -> int:
        return torch.npu.device_count()

    @staticmethod
    @override
    def is_available() -> bool:
        return torch.npu.is_available()
    
    @staticmethod
    @override
    def name() -> str:
        return "NPUAccelerator"

    @override
    def setup(self, trainer: "pl.Trainer") -> None:
        """Called by the Trainer to set up the accelerator."""
        self.set_ascend_flags(trainer.local_rank)
        torch.npu.empty_cache()

    @staticmethod
    def set_ascend_flags(local_rank: int) -> None:
        """Set Ascend NPU environment variables, mirroring CUDA's PCI ordering setup."""
        os.environ["ASCEND_DEVICE_ID"] = str(local_rank)

        all_npu_ids = ",".join(str(x) for x in range(torch.npu.device_count()))
        devices = os.getenv("ASCEND_RT_VISIBLE_DEVICES", all_npu_ids)
        _log.info(f"LOCAL_RANK: {local_rank} - ASCEND_RT_VISIBLE_DEVICES: [{devices}]")

    @override
    def get_device_stats(self, device: Union[str, torch.device]) -> Dict[str, Any]:
        """Return NPU memory stats."""
        if isinstance(device, str):
            device = torch.device(device)
        try:
            return torch_npu.npu.memory_stats(device)
        except Exception:
            return {}

    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry) -> None:
        accelerator_registry.register(
            "npu",
            cls,
            description="NPU Accelerator - optimized for large-scale machine learning.",
        )
