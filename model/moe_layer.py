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
        expert_chunk_size: int = 2,
    ) -> None:
        if expert_chunk_size <= 0:
            raise ValueError("expert_chunk_size must be positive")

        self.layer_id = layer_id
        self.router = router
        self.cache_manager = cache_manager
        self.model_loader = model_loader
        self.transfer_scheduler = transfer_scheduler
        self.expert_chunk_size = expert_chunk_size
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
        """Run one Mixtral MoE layer."""

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

        expert_indices, expert_weights = self.router.route(
            hidden_states
        )

        output = torch.zeros_like(hidden_states)

        # Route once for the full token set, then process the selected experts
        # in bounded groups. This preserves exact per-token top-2 routing while
        # preventing all experts touched by the prompt from accumulating in the
        # GPU working set.
        unique_expert_ids = sorted(
            int(expert_id)
            for expert_id in expert_indices.reshape(-1).unique().tolist()
        )

        for chunk_start in range(0, len(unique_expert_ids), self.expert_chunk_size):
            expert_chunk = unique_expert_ids[
                chunk_start : chunk_start + self.expert_chunk_size
            ]

            # Experts already resident before this forward are left cached for
            # reuse. Experts brought in specifically for this chunk are
            # released afterwards, so completed chunks do not accumulate on GPU.
            was_gpu_resident = {
                expert_id: self.cache_manager.is_gpu_resident(
                    self.layer_id,
                    expert_id,
                )
                for expert_id in expert_chunk
            }

            try:
                for expert_id in expert_chunk:
                    expert = self._ensure_expert_on_gpu(expert_id)

                    # A routed expert can appear in either top-k slot. Build a
                    # single token mask and effective routing weight so each
                    # expert executes its assigned tokens exactly once.
                    expert_matches = expert_indices == expert_id
                    token_mask = expert_matches.any(dim=-1)

                    if not token_mask.any():
                        del expert
                        continue

                    token_weights = (
                        expert_weights * expert_matches.to(expert_weights.dtype)
                    ).sum(dim=-1)

                    expert_input = hidden_states[token_mask]
                    expert_output = expert(expert_input)
                    weighted_output = (
                        expert_output
                        * token_weights[token_mask].unsqueeze(-1)
                    )

                    output[token_mask] += weighted_output

                    # Drop temporary references before moving on so completed
                    # experts can actually release their GPU allocations when
                    # their cache entry is removed below.
                    del expert
                    del expert_input
                    del expert_output
                    del weighted_output
                    del expert_matches
                    del token_mask
                    del token_weights
            finally:
                for expert_id in expert_chunk:
                    if not was_gpu_resident[expert_id]:
                        self.cache_manager.demote_to_cpu(
                            self.layer_id,
                            expert_id,
                        )

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