from __future__ import annotations

import logging
import time
from typing import Optional

import torch

from cache.manager import CacheManager
from model.loader import ModelLoader
from engine.router import ExpertRouter
from transfer.scheduler import TransferScheduler
from transfer.types import (
    TransferDirection,
    TransferPriority,
    TransferRequest,
)


class MoELayer:
    """Single Mixtral MoE layer with expert offloading."""

    def __init__(
        self,
        layer_id: int,
        router: ExpertRouter,
        cache_manager: CacheManager,
        model_loader: ModelLoader,
        transfer_scheduler: TransferScheduler,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.layer_id = layer_id
        self.router = router
        self.cache_manager = cache_manager
        self.model_loader = model_loader
        self.transfer_scheduler = transfer_scheduler
        self._logger = logger or logging.getLogger(__name__)

    def _ensure_expert_on_gpu(
        self,
        expert_id: int,
    ):
        """Return an expert that is resident on GPU."""

        expert = self.cache_manager.get(
            self.layer_id,
            expert_id,
        )

        # Cache miss: load the quantized expert into CPU cache.
        if expert is None:
            self._logger.debug(
                "Expert miss: layer=%d expert=%d",
                self.layer_id,
                expert_id,
            )

            expert = self.model_loader.load_expert(
                self.layer_id,
                expert_id,
            )

            self.cache_manager.put(
                self.layer_id,
                expert_id,
                expert,
                device="cpu",
            )

        # Already on GPU.
        if expert.device.type == "cuda":
            return expert

        # CPU-resident: transfer through the scheduler.
        request = TransferRequest(
            request_id=(
                f"layer-{self.layer_id}-expert-{expert_id}"
                f"-{id(self)}"
            ),
            layer_id=self.layer_id,
            expert_id=expert_id,
            direction=TransferDirection.CPU_TO_GPU,
            priority=TransferPriority.HIGH,
            issued_at=time.monotonic(),
        )

        submitted = self.transfer_scheduler.submit(request)

        if not submitted:
            raise RuntimeError(
                f"Failed to submit transfer for "
                f"layer={self.layer_id}, expert={expert_id}"
            )

        result = self.transfer_scheduler.execute_next()

        if result is None or not result.success:
            error = result.error if result is not None else "no transfer result"
            raise RuntimeError(
                f"Failed to transfer expert "
                f"layer={self.layer_id}, expert={expert_id}: {error}"
            )

        expert = self.cache_manager.get(
            self.layer_id,
            expert_id,
        )

        if expert is None or expert.device.type != "cuda":
            raise RuntimeError(
                f"Expert layer={self.layer_id}, expert={expert_id} "
                "was not resident on GPU after transfer"
            )

        return expert

    @torch.no_grad()
    def _forward_tokens(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Run one Mixtral MoE layer with a full-layer expert working set.

        This is the no-chunk baseline. Routing is computed once for the whole
        token batch, all unique routed experts for this layer are made resident
        on GPU, each expert processes all of its assigned tokens exactly once,
        and experts that were not already GPU-resident are demoted afterward.
        """

        if hidden_states.ndim != 2:
            raise ValueError(
                "hidden_states must have shape "
                "(tokens, hidden_size)"
            )

        if hidden_states.shape[-1] != self.router.hidden_size:
            raise ValueError(
                f"hidden size {hidden_states.shape[-1]} != "
                f"router hidden size {self.router.hidden_size}"
            )

        # One routing pass for the entire token set.
        expert_indices, expert_weights = self.router.route(
            hidden_states
        )

        output = torch.zeros_like(hidden_states)

        # Exact expert working set for this layer. Worst case: all 8 experts.
        unique_expert_ids = sorted(
            int(expert_id)
            for expert_id in expert_indices.reshape(-1).unique().tolist()
        )

        was_gpu_resident = {
            expert_id: self.cache_manager.is_gpu_resident(
                self.layer_id,
                expert_id,
            )
            for expert_id in unique_expert_ids
        }

        gpu_experts = {}
        try:
            # Ensure every expert needed by this layer is resident before
            # beginning expert execution. No chunking.
            for expert_id in unique_expert_ids:
                gpu_experts[expert_id] = self._ensure_expert_on_gpu(expert_id)

            # Each expert processes all tokens routed to it in one batched
            # invocation. An expert may appear in either top-k slot.
            for expert_id in unique_expert_ids:
                expert = gpu_experts[expert_id]
                expert_matches = expert_indices == expert_id
                token_mask = expert_matches.any(dim=-1)

                if not token_mask.any():
                    continue

                token_weights = (
                    expert_weights
                    * expert_matches.to(expert_weights.dtype)
                ).sum(dim=-1)

                expert_input = hidden_states[token_mask]
                expert_output = expert(expert_input)
                weighted_output = (
                    expert_output
                    * token_weights[token_mask].unsqueeze(-1)
                )
                output[token_mask] += weighted_output

                del expert_input
                del expert_output
                del weighted_output
                del expert_matches
                del token_mask
                del token_weights

        finally:
            # Keep the current cache semantics for this experiment. Experts
            # that were not resident before this layer are released after use.
            for expert_id in unique_expert_ids:
                if not was_gpu_resident[expert_id]:
                    self.cache_manager.demote_to_cpu(
                        self.layer_id,
                        expert_id,
                    )
            gpu_experts.clear()

        return output

    def forward(self, hidden_states):

        if hidden_states.ndim == 3:
            batch_size, sequence_length, hidden_size = hidden_states.shape

            hidden_states = hidden_states.reshape(
                batch_size * sequence_length,
                hidden_size,
            )

            output = self._forward_tokens(hidden_states)

            return output.reshape(
                batch_size,
                sequence_length,
                hidden_size,
            )

        if hidden_states.ndim == 2:
            return self._forward_tokens(hidden_states)

        raise ValueError(
        "hidden_states must have shape "
        "(tokens, hidden_size) or (batch, sequence, hidden_size)"
    )