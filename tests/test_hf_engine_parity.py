from __future__ import annotations

import gc
import os

import pytest
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, MixtralForCausalLM

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

PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 8


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required",
)
def test_engine_matches_hf_greedy_generation() -> None:
    """Compare the first generated tokens against HF Mixtral.generate().

    HF is loaded separately on both T4s in 4-bit NF4 mode so the reference
    can fit in the Kaggle dual-T4 environment. MoEInfra then runs on cuda:0
    with its offloaded expert store.

    Important: HF quantizes all Linear modules here, while MoEInfra keeps the
    dense projections in fp16 and only the experts in NF4. Therefore a token
    mismatch is a signal to investigate, not by itself proof of an engine bug.
    """

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    inputs = tokenizer(
        PROMPT,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids_cpu = inputs["input_ids"]
    prompt_len = input_ids_cpu.shape[1]

    # ---------- HuggingFace reference ----------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    hf_device_map = {
        "model.embed_tokens": 0,
        "model.rotary_emb": 0,

        **{f"model.layers.{i}": 0 for i in range(15)},
        **{f"model.layers.{i}": 1 for i in range(15, 32)},

        "model.norm": 1,
        "lm_head": 1,
    }

    hf_model = MixtralForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        device_map=hf_device_map,
        quantization_config=bnb_config,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    hf_model.eval()

    hf_inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

    with torch.inference_mode():
        hf_output = hf_model.generate(
            **hf_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            return_dict_in_generate=True,
        )

    hf_generated = hf_output.sequences[0, prompt_len:].tolist()

    # Release both GPUs before starting MoEInfra.
    del hf_output, hf_inputs, hf_model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # ---------- MoEInfra ----------
    config = load_config("config.yaml")
    config["model"]["name"] = MODEL_PATH
    config["model"]["device"] = "cuda:0"
    config["model"]["dtype"] = "float16"
    config["model"]["num_layers"] = 32
    config["model"]["expert_store_paths"] = [STORE1, STORE2]
    config.setdefault("cache", {})["gpu_slots"] = 1
    config["cache"]["cpu_slots"] = 8

    engine = InferenceEngine(config)

    try:
        engine_result = engine.generate(
            ForwardRequest(
                input_ids=input_ids_cpu.to("cuda:0"),
                max_new_tokens=MAX_NEW_TOKENS,
            )
        )
        engine_generated = list(engine_result.token_ids)

        n_compare = min(len(hf_generated), len(engine_generated), MAX_NEW_TOKENS)
        hf_prefix = hf_generated[:n_compare]
        engine_prefix = engine_generated[:n_compare]

        print(f"PROMPT: {PROMPT!r}")
        print(f"PROMPT TOKENS: {input_ids_cpu[0].tolist()}")
        print(f"HF TOKENS:     {hf_generated}")
        print(f"ENGINE TOKENS: {engine_generated}")
        print(
            "HF TEXT:       ",
            repr(tokenizer.decode(hf_generated, skip_special_tokens=False)),
        )
        print(
            "ENGINE TEXT:   ",
            repr(tokenizer.decode(engine_generated, skip_special_tokens=False)),
        )

        mismatches = [
            i
            for i, (hf_id, engine_id) in enumerate(zip(hf_prefix, engine_prefix))
            if hf_id != engine_id
        ]

        assert n_compare == MAX_NEW_TOKENS, (
            f"Could only compare {n_compare} generated tokens; "
            f"HF={len(hf_generated)}, engine={len(engine_generated)}"
        )
        assert not mismatches, (
            f"Greedy token mismatch at generated positions {mismatches}. "
            f"HF={hf_prefix}, engine={engine_prefix}"
        )
    finally:
        engine.shutdown()
        del engine
        gc.collect()
        torch.cuda.empty_cache()
