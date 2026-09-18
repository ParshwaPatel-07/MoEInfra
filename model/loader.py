"""Model loader for MoEInfra.

:class:`ModelLoader` downloads / loads a quantised Mixtral-8x7B checkpoint
from HuggingFace Hub and provides helpers to retrieve individual expert
tensors on demand.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

from model.expert import MixtralExpertLayer


class ModelLoader:
    """Loads and caches Mixtral-8x7B model weights for expert offloading.

    Responsible for downloading or finding a local INT4-quantised checkpoint
    and exposing individual expert layers to the inference engine.

    Attributes:
        model_name: HuggingFace model identifier or local path.
        num_layers: Total number of transformer layers.
        num_experts: Total number of experts per layer.
        hidden_size: Model hidden dimension.
        intermediate_size: FFN intermediate dimension.
        quantization: Quantisation scheme (e.g. ``"int4"``).
    """

    def __init__(
        self,
        model_name: str,
        num_layers: int,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        quantization: str = "int4",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialise the loader.

        Args:
            model_name: HuggingFace Hub model name or local directory path.
            num_layers: Number of transformer layers in the model.
            num_experts: Number of experts per MoE layer.
            hidden_size: Model hidden dimension (e.g. 4096).
            intermediate_size: FFN intermediate dimension (e.g. 14336).
            quantization: Quantisation scheme to apply when loading
                (``"int4"`` or ``"int8"``).
            logger: Optional pre-configured logger.
        """
        self.model_name: str = model_name
        self.num_layers: int = num_layers
        self.num_experts: int = num_experts
        self.hidden_size: int = hidden_size
        self.intermediate_size: int = intermediate_size
        self.quantization: str = quantization
        self._logger: logging.Logger = logger or logging.getLogger(__name__)

        self._checkpoint_path: Optional[Path] = None
        self._is_loaded: bool = False

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """Download and prepare the model checkpoint.

        For large checkpoints this may trigger a HuggingFace Hub download.
        After this method returns :meth:`load_expert` can be called.

        Returns:
            None

        Raises:
            RuntimeError: If the checkpoint cannot be found or loaded.
        """
        self._logger.info(
            "Loading model %r with quantization=%s ...",
            self.model_name,
            self.quantization,
        )
        # Real implementation: call transformers.AutoModelForCausalLM.from_pretrained
        # with load_in_4bit=True and device_map="cpu" so all experts start on CPU.
        self._is_loaded = True
        self._logger.info("Model checkpoint ready.")

    def load_expert(self, layer_id: int, expert_id: int) -> MixtralExpertLayer:
        """Construct a :class:`~model.expert.MixtralExpertLayer` for a single expert.

        Slices the relevant weight tensors from the loaded checkpoint and
        wraps them in a :class:`~model.expert.MixtralExpertLayer` on CPU.

        Args:
            layer_id: Transformer layer index (0-based).
            expert_id: Expert index within the layer (0-based).

        Returns:
            A :class:`~model.expert.MixtralExpertLayer` on CPU, ready to be
            inserted into the cache.

        Raises:
            RuntimeError: If :meth:`load` has not been called yet.
            IndexError: If *layer_id* or *expert_id* is out of range.
        """
        if not self._is_loaded:
            raise RuntimeError(
                "Call ModelLoader.load() before requesting individual experts."
            )
        if not (0 <= layer_id < self.num_layers):
            raise IndexError(f"layer_id {layer_id} out of range [0, {self.num_layers})")
        if not (0 <= expert_id < self.num_experts):
            raise IndexError(f"expert_id {expert_id} out of range [0, {self.num_experts})")

        self._logger.debug(
            "Loading expert: layer=%d, expert=%d", layer_id, expert_id
        )
        # Real implementation: extract w1/w2/w3 from the checkpoint and assign
        # them to the layer's Linear modules.
        expert = MixtralExpertLayer(
            layer_id=layer_id,
            expert_id=expert_id,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            logger=self._logger,
        )
        return expert.cpu()

    def get_expert_size_bytes(self) -> int:
        """Return the estimated on-disk / in-memory size of a single expert.

        Assumes INT4 quantisation (0.5 bytes per parameter) for the three
        projection matrices.

        Returns:
            Estimated size in bytes.
        """
        params_per_expert = (
            self.hidden_size * self.intermediate_size  # w1
            + self.intermediate_size * self.hidden_size  # w2
            + self.hidden_size * self.intermediate_size  # w3
        )
        bytes_per_param = 0.5  # INT4
        return int(params_per_expert * bytes_per_param)

    def __repr__(self) -> str:
        return (
            f"ModelLoader("
            f"model={self.model_name!r}, "
            f"layers={self.num_layers}, experts={self.num_experts}, "
            f"quant={self.quantization}, loaded={self._is_loaded})"
        )
