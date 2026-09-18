"""Type definitions for the MoEInfra transfer subsystem.

Defines the data structures used to describe PCIe transfers between CPU RAM
and GPU VRAM: :class:`TransferDirection`, :class:`TransferPriority`,
:class:`TransferRequest`, and :class:`TransferResult`.
"""
from __future__ import annotations

import dataclasses
import enum
from typing import Optional

import torch


class TransferDirection(enum.Enum):
    """Direction of a PCIe expert-weight transfer.

    Attributes:
        CPU_TO_GPU: Load an expert from CPU RAM into GPU VRAM.
        GPU_TO_CPU: Evict an expert from GPU VRAM back to CPU RAM.
    """

    CPU_TO_GPU = "cpu_to_gpu"
    GPU_TO_CPU = "gpu_to_cpu"


class TransferPriority(enum.Enum):
    """Scheduling priority for a :class:`TransferRequest`.

    Attributes:
        HIGH: Process before NORMAL and LOW requests.
        NORMAL: Standard scheduling priority.
        LOW: Background / prefetch transfers processed last.
    """

    HIGH = 0
    NORMAL = 1
    LOW = 2


@dataclasses.dataclass
class TransferRequest:
    """A single pending expert-weight transfer request.

    Attributes:
        request_id: Unique identifier (e.g. a UUID string) for this request.
        layer_id: Transformer layer the expert belongs to.
        expert_id: Expert index within the layer.
        direction: Whether the transfer is CPU→GPU or GPU→CPU.
        priority: Scheduling priority for queue ordering.
        issued_at: UNIX timestamp when the request was created.
        tensor: The weight tensor to move, or ``None`` if the scheduler must
            fetch it from the cache manager.
    """

    request_id: str
    layer_id: int
    expert_id: int
    direction: TransferDirection
    priority: TransferPriority
    issued_at: float
    tensor: Optional[torch.Tensor] = None


@dataclasses.dataclass
class TransferResult:
    """The outcome of an executed :class:`TransferRequest`.

    Attributes:
        request_id: Matches :attr:`TransferRequest.request_id`.
        success: ``True`` if the transfer completed without error.
        elapsed_ms: Wall-clock time the transfer took in milliseconds.
        bytes_transferred: Number of bytes moved across the PCIe bus.
        error: Human-readable error message, or ``None`` on success.
    """

    request_id: str
    success: bool
    elapsed_ms: float
    bytes_transferred: int
    error: Optional[str] = None
