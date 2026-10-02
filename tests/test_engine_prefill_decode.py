from __future__ import annotations

import os

import pytest
import torch

from config_loader import load_config
from engine.engine import InferenceEngine
from engine.types import ForwardRequest


MODEL_PATH = (
    "/kaggle/input/models/mistral-ai/mixtral/"
    "pytorch/8x7b-instruct-v0.1-hf/1"
)
STORE1 = (
    "/kaggle/input/datasets/parshwapatel07/"
    "mixtral-8x7b-nf4-experts/experts"
)
STORE2 = (
    "/kaggle/input/datasets/parshwapatel07/"
    "mixtral-8x7b-nf4-experts-part2/experts"
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_real_prefill_and_decode_smoke() -> None:
    config = load_config("config.yaml")
    config["model"]["name"] = MODEL_PATH
    config["model"]["device"] = "cuda"
    config["model"]["dtype"] = "float16"
    config["model"]["expert_store_paths"] = [STORE1, STORE2]
    config["model"]["num_layers"] = min(
        int(os.environ.get("MOEINFRA_PROFILE_LAYERS", "4")),
        32,
    )

    config.setdefault("cache", {})["gpu_slots"] = 1
    config["cache"]["cpu_slots"] = 8

    engine = InferenceEngine(config)

    prompt_len = 4
    max_new_tokens = 4
    input_ids = torch.tensor(
        [[1, 42, 314, 271]],
        dtype=torch.long,
        device="cuda",
    )

    result = engine.generate(
        ForwardRequest(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
        )
    )

    assert len(result.token_ids) == max_new_tokens
    assert result.logits.ndim == 3
    assert result.logits.shape[0] == 1
    assert result.logits.shape[1] == 1
    assert result.logits.shape[2] == engine._mixtral_config.vocab_size
    assert result.logits.device.type == "cuda"
    assert torch.isfinite(result.logits).all()

    assert engine._kv_cache is not None
    assert engine._kv_cache.get_seq_length() == prompt_len + max_new_tokens - 1
    assert engine._attention_mask is not None
    assert engine._attention_mask.shape == (
        1,
        prompt_len + max_new_tokens - 1,
    )

    stats = engine._cache_manager.stats()
    assert stats.gpu_slots_used <= 1
    assert stats.misses > 0

    assert result.ttft_ms > 0
    assert result.stage_timings["prefill"] > 0
    assert result.stage_timings["decode"] >= 0

    engine.shutdown()
