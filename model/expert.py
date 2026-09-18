"""Mixtral expert layer wrapper for MoEInfra.

:class:`MixtralExpertLayer` wraps a single MoE expert's weight tensors and
exposes a :meth:`forward` method plus helpers to move the expert between GPU
and CPU memory.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn


class MixtralExpertLayer(nn.Module):
    """A single INT4-quantised Mixtral FFN expert.

    Holds the ``w1``, ``w2``, and ``w3`` projection matrices that make up a
    SwiGLU expert, and provides helpers to move them between devices.

    Attributes:
        layer_id: Transformer layer this expert belongs to.
        expert_id: Expert index within the layer.
        hidden_size: Input/output hidden dimension.
        intermediate_size: Intermediate FFN dimension.
    """

    def __init__(
        self,
        layer_id: int,
        expert_id: int,
        hidden_size: int,
        intermediate_size: int,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialise the expert layer.

        Args:
            layer_id: Transformer layer index.
            expert_id: Expert index within the layer.
            hidden_size: Model hidden dimension (e.g. 4096).
            intermediate_size: FFN intermediate dimension (e.g. 14336).
            logger: Optional pre-configured logger.
        """
        super().__init__()
        self.layer_id: int = layer_id
        self.expert_id: int = expert_id
        self.hidden_size: int = hidden_size
        self.intermediate_size: int = intermediate_size
        self._logger: logging.Logger = logger or logging.getLogger(__name__)

        # SwiGLU expert projections (unquantised placeholders;
        # real loading via ModelLoader populates INT4 tensors)
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def device(self) -> torch.device:
        """Return the device the expert parameters currently reside on.

        Returns:
            A :class:`torch.device` instance.
        """
        return next(self.parameters()).device

    @property
    def is_on_gpu(self) -> bool:
        """Return ``True`` if the expert is resident in GPU memory.

        Returns:
            ``True`` when :attr:`device` is a CUDA device.
        """
        return self.device.type == "cuda"

    @property
    def size_bytes(self) -> int:
        """Return the total memory footprint of the expert in bytes.

        Returns:
            Sum of bytes used by all parameter tensors.
        """
        return sum(
            p.numel() * p.element_size() for p in self.parameters()
        )

    # ------------------------------------------------------------------ #
    # Forward pass                                                         #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the SwiGLU expert output for input *x*.

        Implements the standard Mixtral SwiGLU FFN:
        ``output = w2(SiLU(w1(x)) * w3(x))``

        Args:
            x: Input tensor of shape ``(tokens, hidden_size)``.

        Returns:
            Output tensor of shape ``(tokens, hidden_size)``.
        """
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))

    # ------------------------------------------------------------------ #
    # Device helpers                                                       #
    # ------------------------------------------------------------------ #

    def to_cpu(self) -> "MixtralExpertLayer":
        """Move this expert to CPU RAM.

        Returns:
            *self* (for chaining).
        """
        self._logger.debug(
            "Expert (layer=%d, expert=%d): GPU → CPU", self.layer_id, self.expert_id
        )
        return self.cpu()

    def to_gpu(self, device: str = "cuda:0") -> "MixtralExpertLayer":
        """Move this expert to GPU memory.

        Args:
            device: CUDA device string (default ``"cuda:0"``).

        Returns:
            *self* (for chaining).
        """
        self._logger.debug(
            "Expert (layer=%d, expert=%d): CPU → %s", self.layer_id, self.expert_id, device
        )
        return self.to(device)

    def __repr__(self) -> str:
        return (
            f"MixtralExpertLayer("
            f"layer={self.layer_id}, expert={self.expert_id}, "
            f"device={self.device}, size={self.size_bytes // 1024}KB)"
        )
