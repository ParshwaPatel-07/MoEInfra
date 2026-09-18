"""Expert router for Mixtral-style MoE inference.

:class:`ExpertRouter` wraps the top-k gating logic and provides a heuristic
helper to predict which experts the *next* layer is likely to need, enabling
the :class:`~engine.prefetch.PrefetchEngine` to issue proactive transfers.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F


class ExpertRouter:
    """Softmax top-k expert router for Mixtral-8x7B.

    Computes per-token expert assignments from raw router logits and provides
    a lightweight heuristic for predicting the next layer's expert set.

    Attributes:
        num_experts: Total number of experts per layer.
        num_experts_per_tok: How many experts each token is routed to (top-k).
    """

    def __init__(
        self,
        num_experts: int,
        num_experts_per_tok: int,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialise the router.

        Args:
            num_experts: Total experts available per layer (e.g. 8 for
                Mixtral-8x7B).
            num_experts_per_tok: Number of experts each token is assigned to
                (e.g. 2 for Mixtral-8x7B).
            logger: Optional pre-configured logger.
        """
        if num_experts_per_tok > num_experts:
            raise ValueError(
                f"num_experts_per_tok ({num_experts_per_tok}) must be <= "
                f"num_experts ({num_experts})"
            )
        self.num_experts: int = num_experts
        self.num_experts_per_tok: int = num_experts_per_tok
        self._logger: logging.Logger = logger or logging.getLogger(__name__)

    def route(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple:
        """Compute top-k expert assignments for a batch of tokens.

        Applies softmax to *router_logits* and selects the top-k experts per
        token.

        Args:
            hidden_states: Hidden-state tensor of shape
                ``(batch * seq_len, hidden_size)``.  Not used in the routing
                computation itself but validates input consistency.
            router_logits: Raw router logit tensor of shape
                ``(batch * seq_len, num_experts)``.

        Returns:
            A tuple ``(expert_indices, expert_weights)`` where:

            - ``expert_indices``: :class:`torch.Tensor` of shape
              ``(batch * seq_len, num_experts_per_tok)`` — int64 indices.
            - ``expert_weights``: :class:`torch.Tensor` of shape
              ``(batch * seq_len, num_experts_per_tok)`` — normalised weights.

        Raises:
            ValueError: If ``router_logits`` has an unexpected second dimension.
        """
        if router_logits.shape[-1] != self.num_experts:
            raise ValueError(
                f"router_logits last dim {router_logits.shape[-1]} != "
                f"num_experts {self.num_experts}"
            )

        routing_weights = F.softmax(router_logits, dim=-1)
        expert_weights, expert_indices = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
        )
        # Re-normalise so weights sum to 1 per token
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        return expert_indices, expert_weights

    def predict_next_layer_experts(
        self, current_expert_indices: torch.Tensor
    ) -> list:
        """Heuristic: predict which experts the next layer will need.

        Uses the set of expert IDs active in the *current* layer as a proxy
        for the next layer.  This simple heuristic avoids the overhead of a
        look-ahead forward pass while still covering the common case where
        token-to-expert assignments stay locally consistent.

        Args:
            current_expert_indices: Int tensor of shape
                ``(tokens, num_experts_per_tok)`` from the current layer's
                :meth:`route` call.

        Returns:
            Sorted list of unique expert IDs predicted to be needed next.
        """
        flat = current_expert_indices.reshape(-1)
        unique_ids = flat.unique().tolist()
        return sorted(int(x) for x in unique_ids)
