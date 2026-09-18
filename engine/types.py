"""Type definitions for the MoEInfra inference engine.

Defines :class:`PrefetchPolicy`, :class:`ForwardRequest`, and
:class:`ForwardResult` used throughout the engine subsystem.
"""
from __future__ import annotations

import dataclasses
import enum
from typing import Optional

import torch


class PrefetchPolicy(enum.Enum):
    """Strategy for prefetching expert weights ahead of compute.

    Attributes:
        NONE: No prefetching; experts are loaded synchronously on demand.
        NEXT_LAYER: Prefetch experts for the immediately following layer only.
        LOOKAHEAD: Prefetch experts for the next *depth* layers (configurable
            via :attr:`~engine.prefetch.PrefetchEngine.depth`).
    """

    NONE = "none"
    NEXT_LAYER = "next_layer"
    LOOKAHEAD = "lookahead"


@dataclasses.dataclass
class ForwardRequest:
    """A single inference request to the :class:`~engine.engine.InferenceEngine`.

    Attributes:
        input_ids: Token ID tensor of shape ``(batch, seq_len)``.
        attention_mask: Boolean mask of shape ``(batch, seq_len)``, or
            ``None`` to use a default all-ones mask.
        max_new_tokens: Maximum number of tokens to generate in decode phase.
    """

    input_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    max_new_tokens: int = 128


@dataclasses.dataclass
class ForwardResult:
    """The result of a completed :class:`ForwardRequest`.

    Attributes:
        logits: Final logit tensor of shape ``(batch, vocab_size)``.
        token_ids: Flat list of generated token IDs (decode phase output).
        ttft_ms: Time-to-first-token in milliseconds (prefill latency).
        tps: Tokens per second during the decode phase.
        stage_timings: Dict mapping stage names to elapsed milliseconds,
            e.g. ``{"compute": 42.1, "attention": 12.3, "router": 1.5,
            "sync": 0.8}``.
    """

    logits: torch.Tensor
    token_ids: list
    ttft_ms: float
    tps: float
    stage_timings: dict
