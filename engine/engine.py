"""Top-level inference engine for MoEInfra.

:class:`InferenceEngine` orchestrates the full prefill + decode pipeline:
it loads the model, manages the cache and prefetch subsystems, measures
latency, and returns a :class:`~engine.types.ForwardResult`.

The :meth:`InferenceEngine._prefill` and :meth:`InferenceEngine._decode`
methods are stubs — they must be implemented once the model-loading and
offloading layers are in place.
"""
from __future__ import annotations

import logging
import time
import types
from typing import Optional

import torch
import torch.nn as nn
from transformers import MixtralConfig
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.models.mixtral.modeling_mixtral import (
    MixtralRMSNorm,
    MixtralRotaryEmbedding,
)

from cache.manager import CacheManager
from cache.types import EvictionPolicy
from engine.decoder_layer import MoEInfraDecoderLayer
from model.loader import ModelLoader
from model.moe_layer import MoELayer

from engine.types import ForwardRequest, ForwardResult, PrefetchPolicy
from engine.router import ExpertRouter
from engine.prefetch import PrefetchEngine
from transfer.pinned_memory import PinnedMemoryBudget
from transfer.reusable_staging import ReusablePinnedStagingPool
from transfer.scheduler import TransferScheduler


class InferenceEngine:
    """End-to-end INT4 Mixtral-8x7B inference engine with expert offloading.

    Wraps the model loader, cache manager, transfer scheduler, expert router,
    and prefetch engine into a single high-level interface.  Provides a
    context-manager protocol for clean resource teardown.

    Attributes:
        config: Full configuration dict loaded from ``config.yaml``.
    """

    def __init__(
        self,
        config: dict,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialise the engine and all subsystems.

        Args:
            config: Full configuration dict as returned by
                :func:`~config_loader.load_config`.
            logger: Optional pre-configured logger.
        """
        self.config: dict = config
        self._logger: logging.Logger = logger or logging.getLogger(__name__)

        model_cfg: dict = config.get("model", {})
        cache_cfg: dict = config.get("cache", {})
        transfer_cfg: dict = config.get("transfer", {})

        self._num_experts: int = model_cfg.get("num_experts", 8)
        self._num_experts_per_tok: int = model_cfg.get("num_experts_per_tok", 2)
        self._num_layers: int = model_cfg.get("num_layers", 32)

        # Persistent Mixtral runtime configuration. The actual weights and
        # decoder stack are built lazily once, on the first inference call,
        # so constructing InferenceEngine itself remains lightweight.
        self._device = torch.device(
            model_cfg.get(
                "device",
                "cuda" if torch.cuda.is_available() else "cpu",
            )
        )

        dtype_name = model_cfg.get(
            "dtype",
            "float16" if self._device.type == "cuda" else "float32",
        ).lower()
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if dtype_name not in dtype_map:
            raise ValueError(
                f"Unsupported model dtype {dtype_name!r}; "
                f"expected one of {sorted(dtype_map)}"
            )
        self._dtype = dtype_map[dtype_name]

        self._mixtral_config = MixtralConfig.from_pretrained(
            model_cfg.get("name")
        )
        self._mixtral_config._attn_implementation = model_cfg.get(
            "attn_implementation",
            "eager",
        )

        self._decoder_layers: nn.ModuleList = nn.ModuleList()
        self._embed_tokens: Optional[nn.Embedding] = None
        self._final_norm: Optional[MixtralRMSNorm] = None
        self._lm_head: Optional[nn.Linear] = None
        self._rotary_emb: Optional[MixtralRotaryEmbedding] = None

        # Persistent generation state. Prefill creates the DynamicCache;
        # decode reuses and extends the same cache.
        self._kv_cache: Optional[DynamicCache] = None
        self._attention_mask: Optional[torch.Tensor] = None
        self._next_token_logits: Optional[torch.Tensor] = None
        self._last_logits: Optional[torch.Tensor] = None
        self._runtime_initialized = False

        # Lazy-initialised subsystems (require stub implementations)
       # ── Core subsystems ─────────────────────────────────────────────── #

        self._cache_manager = CacheManager(
            gpu_slots=cache_cfg.get("gpu_slots", 8),
            cpu_slots=cache_cfg.get("cpu_slots", 32),
            policy=EvictionPolicy(cache_cfg.get("eviction_policy", "lru")),
            logger=self._logger,
        )

        staging_budget = PinnedMemoryBudget(
            budget_bytes=transfer_cfg.get(
                "staging_budget_mb",
                512,
            ) * 1024 * 1024,
        )
        self._staging_pool = ReusablePinnedStagingPool(
            budget=staging_budget,
            slot_size_bytes=transfer_cfg.get(
                "staging_slot_mb",
                128,
            ) * 1024 * 1024,
        )
        self._transfer_stream = (
            torch.cuda.Stream()
            if self._device.type == "cuda"
            else None
        )

        self._transfer_scheduler = TransferScheduler(
            cache_manager=self._cache_manager,
            bandwidth_gbps=transfer_cfg.get("bandwidth_gbps", 8.0),
            max_concurrent=transfer_cfg.get("max_concurrent_transfers", 4),
            staging_pool=self._staging_pool,
            transfer_stream=self._transfer_stream,
            logger=self._logger,
        )

        self._prefetch_engine = PrefetchEngine(
            cache_manager=self._cache_manager,
            transfer_scheduler=self._transfer_scheduler,
            policy=PrefetchPolicy.NONE,
            depth=transfer_cfg.get("prefetch_depth", 2),
            logger=self._logger,
        )

        self._model = ModelLoader(
            model_name=model_cfg.get("name"),
            num_layers=self._num_layers,
            num_experts=self._num_experts,
            hidden_size=model_cfg.get("hidden_size", 4096),
            intermediate_size=model_cfg.get("intermediate_size", 14336),
            quantization=model_cfg.get("quantization", "int4"),
            logger=self._logger,
            expert_store_paths=model_cfg.get("expert_store_paths"),
        )

        self._logger.info(
            "InferenceEngine initialised — model=%s, layers=%d, experts=%d/%d",
            model_cfg.get("name", "unknown"),
            self._num_layers,
            self._num_experts_per_tok,
            self._num_experts,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _initialize_runtime(self) -> None:
        """Build the persistent Mixtral execution graph exactly once.

        The low-level cache, transfer, and expert-loading components are
        reused unchanged. Decoder layers are patched with MoEInfra's
        ``MoELayer`` before they are moved to the target device so the native
        Hugging Face expert weights are never copied to GPU.
        """
        if self._runtime_initialized:
            return

        self._model.load()

        hidden_size = self._mixtral_config.hidden_size
        vocab_size = self._mixtral_config.vocab_size

        self._embed_tokens = nn.Embedding(
            vocab_size,
            hidden_size,
            device=self._device,
            dtype=self._dtype,
        )
        embedding_weight = self._model.load_embeddings()
        self._embed_tokens.weight.copy_(
            embedding_weight.to(device=self._device, dtype=self._dtype)
        )

        self._rotary_emb = MixtralRotaryEmbedding(
            config=self._mixtral_config,
        ).to(self._device)

        for layer_id in range(self._num_layers):
            weights = self._model.load_layer(layer_id)

            # Replace the native Mixtral MoE block before device transfer.
            # The persistent decoder layer then contains only attention,
            # layer norms, and the MoE adapter.
            router = ExpertRouter(
                gate_weight=weights.moe_gate.to(
                    device=self._device,
                    dtype=self._dtype,
                ),
                num_experts_per_tok=self._num_experts_per_tok,
                logger=self._logger,
            )
            moe_layer = MoELayer(
                layer_id=layer_id,
                router=router,
                cache_manager=self._cache_manager,
                model_loader=self._model,
                transfer_scheduler=self._transfer_scheduler,
                logger=self._logger,
            )

            decoder_layer = MoEInfraDecoderLayer(
                self._mixtral_config,
                layer_idx=layer_id,
            )
            decoder_layer.set_moe(moe_layer)
            decoder_layer.to(device=self._device, dtype=self._dtype)

            with torch.no_grad():
                decoder_layer.input_layernorm.weight.copy_(
                    weights.input_layernorm.to(
                        device=self._device,
                        dtype=self._dtype,
                    )
                )
                decoder_layer.self_attn.q_proj.weight.copy_(
                    weights.q_proj.to(
                        device=self._device,
                        dtype=self._dtype,
                    )
                )
                decoder_layer.self_attn.k_proj.weight.copy_(
                    weights.k_proj.to(
                        device=self._device,
                        dtype=self._dtype,
                    )
                )
                decoder_layer.self_attn.v_proj.weight.copy_(
                    weights.v_proj.to(
                        device=self._device,
                        dtype=self._dtype,
                    )
                )
                decoder_layer.self_attn.o_proj.weight.copy_(
                    weights.o_proj.to(
                        device=self._device,
                        dtype=self._dtype,
                    )
                )
                decoder_layer.post_attention_layernorm.weight.copy_(
                    weights.post_attention_layernorm.to(
                        device=self._device,
                        dtype=self._dtype,
                    )
                )

            self._decoder_layers.append(decoder_layer)
            del weights

        self._final_norm = MixtralRMSNorm(
            hidden_size,
            eps=self._mixtral_config.rms_norm_eps,
        ).to(device=self._device, dtype=self._dtype)
        final_norm_weight = self._model.load_final_norm()
        self._final_norm.weight.copy_(
            final_norm_weight.to(device=self._device, dtype=self._dtype)
        )

        self._lm_head = nn.Linear(
            hidden_size,
            vocab_size,
            bias=False,
            device=self._device,
            dtype=self._dtype,
        )
        lm_head_weight = self._model.load_lm_head()
        self._lm_head.weight.copy_(
            lm_head_weight.to(device=self._device, dtype=self._dtype)
        )

        self._kv_cache = None
        self._runtime_initialized = True

        self._logger.info(
            "Persistent Mixtral runtime initialized: layers=%d device=%s dtype=%s",
            len(self._decoder_layers),
            self._device,
            self._dtype,
        )

    def _forward_layers(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the persistent decoder stack for one prefill/decode step."""
        if self._kv_cache is None:
            raise RuntimeError("KV cache is not initialized")
        if self._rotary_emb is None:
            raise RuntimeError("Rotary embedding is not initialized")

        causal_mask = create_causal_mask(
            config=self._mixtral_config,
            input_embeds=hidden_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=self._kv_cache,
            position_ids=position_ids,
        )

        position_embeddings = self._rotary_emb(
            hidden_states,
            position_ids=position_ids,
        )

        for decoder_layer in self._decoder_layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=self._kv_cache,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        return hidden_states

    def generate(self, request: ForwardRequest) -> ForwardResult:
        """Run the full prefill-then-decode pipeline for *request*."""
        self._initialize_runtime()

        stage_timings: dict = {}

        # ── Prefill ───────────────────────────────────────────────────── #
        t0 = time.perf_counter()
        next_token_logits = self._prefill(
            request.input_ids,
            attention_mask=request.attention_mask,
        )
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        ttft_ms = (time.perf_counter() - t0) * 1000.0
        stage_timings["prefill"] = ttft_ms

        # ── Decode ────────────────────────────────────────────────────── #
        t1 = time.perf_counter()
        token_ids = self._decode(next_token_logits, request.max_new_tokens)
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        decode_ms = (time.perf_counter() - t1) * 1000.0
        stage_timings["decode"] = decode_ms

        n_tokens = len(token_ids)
        decode_steps = max(n_tokens - 1, 0)
        tps = (
            decode_steps / (decode_ms / 1000.0)
            if decode_steps > 0 and decode_ms > 0
            else 0.0
        )

        logits = (
            self._last_logits
            if self._last_logits is not None
            else next_token_logits
        )

        return ForwardResult(
            logits=logits,
            token_ids=token_ids,
            ttft_ms=ttft_ms,
            tps=tps,
            stage_timings=stage_timings,
        )

    @torch.no_grad()
    def _prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode the complete prompt and initialize the KV cache."""
        if self._embed_tokens is None or self._final_norm is None or self._lm_head is None:
            raise RuntimeError("Inference runtime has not been initialized")

        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape (batch, seq_len), got {tuple(input_ids.shape)}"
            )

        input_ids = input_ids.to(device=self._device, dtype=torch.long)
        batch_size, sequence_length = input_ids.shape

        if sequence_length <= 0:
            raise ValueError("input_ids must contain at least one token")

        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size,
                sequence_length,
                dtype=torch.long,
                device=self._device,
            )
        else:
            if attention_mask.shape != input_ids.shape:
                raise ValueError(
                    "attention_mask must have the same shape as input_ids"
                )
            attention_mask = attention_mask.to(
                device=self._device,
                dtype=torch.long,
            )

        self._kv_cache = DynamicCache(config=self._mixtral_config)
        self._attention_mask = attention_mask

        cache_position = torch.arange(
            sequence_length,
            device=self._device,
            dtype=torch.long,
        )
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

        hidden_states = self._embed_tokens(input_ids)
        hidden_states = self._forward_layers(
            hidden_states,
            position_ids=position_ids,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
        hidden_states = self._final_norm(hidden_states)

        logits = self._lm_head(hidden_states[:, -1:, :])
        self._next_token_logits = logits
        self._last_logits = logits

        return logits

    @torch.no_grad()
    def _decode(
        self,
        next_token_logits: torch.Tensor,
        max_new_tokens: int,
    ) -> list:
        """Greedily generate tokens while reusing the prefill KV cache."""
        if self._embed_tokens is None or self._final_norm is None or self._lm_head is None:
            raise RuntimeError("Inference runtime has not been initialized")
        if self._kv_cache is None or self._attention_mask is None:
            raise RuntimeError("Prefill must run before decode")

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if max_new_tokens == 0:
            return []
        if next_token_logits.shape[0] != 1:
            raise ValueError("Current decode path supports batch_size=1")

        generated: list[int] = []
        current_logits = next_token_logits

        # The first generated token comes directly from the prompt logits.
        next_token = current_logits[:, -1, :].argmax(dim=-1)
        generated.append(int(next_token.item()))

        # Each subsequent token requires one new decoder pass.
        for _ in range(1, max_new_tokens):
            current_input = next_token.unsqueeze(1)

            self._attention_mask = torch.cat(
                [
                    self._attention_mask,
                    torch.ones(
                        1,
                        1,
                        dtype=self._attention_mask.dtype,
                        device=self._device,
                    ),
                ],
                dim=-1,
            )

            cache_position = torch.tensor(
                [self._kv_cache.get_seq_length()],
                dtype=torch.long,
                device=self._device,
            )
            position_ids = cache_position.unsqueeze(0)

            hidden_states = self._embed_tokens(current_input)
            hidden_states = self._forward_layers(
                hidden_states,
                position_ids=position_ids,
                cache_position=cache_position,
                attention_mask=self._attention_mask,
            )
            hidden_states = self._final_norm(hidden_states)
            current_logits = self._lm_head(hidden_states)

            self._last_logits = current_logits
            next_token = current_logits[:, -1, :].argmax(dim=-1)
            generated.append(int(next_token.item()))

        return generated

    def shutdown(self) -> None:
        """Release GPU/CPU resources and stop background threads.

        Returns:
            None
        """
        self._logger.info("InferenceEngine shutting down.")
        # Real implementation: drain transfer scheduler, free GPU memory, etc.

    # ------------------------------------------------------------------ #
    # Context-manager protocol                                             #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "InferenceEngine":
        """Return *self* for use as a context manager.

        Returns:
            This :class:`InferenceEngine` instance.
        """
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[types.TracebackType],
    ) -> None:
        """Call :meth:`shutdown` on exit.

        Args:
            exc_type: Exception type, if any.
            exc_val: Exception value, if any.
            exc_tb: Traceback, if any.
        """
        self.shutdown()
