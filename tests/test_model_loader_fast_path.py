import json

import pytest
import torch
from safetensors.torch import save_file

from model.loader import ModelLoader
from model.types import LayerWeights
from model.expert import QuantizedMixtralExpert, PrequantizedNF4Expert

HIDDEN = 8
INTERMEDIATE = 16
VOCAB = 12
NUM_LAYERS = 2
NUM_EXPERTS = 2


def make_checkpoint(tmp_path):
    """Create a tiny Mixtral-like checkpoint."""

    tensors = {
        "model.embed_tokens.weight":
            torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16),

        "model.norm.weight":
            torch.randn(HIDDEN, dtype=torch.bfloat16),

        "lm_head.weight":
            torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16),
    }

    for layer in range(NUM_LAYERS):
        prefix = f"model.layers.{layer}"

        tensors.update({
            f"{prefix}.input_layernorm.weight":
                torch.randn(HIDDEN, dtype=torch.bfloat16),

            f"{prefix}.self_attn.q_proj.weight":
                torch.randn(HIDDEN, HIDDEN, dtype=torch.bfloat16),

            f"{prefix}.self_attn.k_proj.weight":
                torch.randn(HIDDEN // 2, HIDDEN, dtype=torch.bfloat16),

            f"{prefix}.self_attn.v_proj.weight":
                torch.randn(HIDDEN // 2, HIDDEN, dtype=torch.bfloat16),

            f"{prefix}.self_attn.o_proj.weight":
                torch.randn(HIDDEN, HIDDEN, dtype=torch.bfloat16),

            f"{prefix}.post_attention_layernorm.weight":
                torch.randn(HIDDEN, dtype=torch.bfloat16),

            f"{prefix}.block_sparse_moe.gate.weight":
                torch.randn(NUM_EXPERTS, HIDDEN, dtype=torch.bfloat16),
        })

    shard_name = "model-00001-of-00001.safetensors"

    save_file(
        tensors,
        str(tmp_path / shard_name),
    )

    weight_map = {
        name: shard_name
        for name in tensors
    }

    with (tmp_path / "model.safetensors.index.json").open("w") as f:
        json.dump({"weight_map": weight_map}, f)

    return tensors


def make_loader(tmp_path):
    return ModelLoader(
        model_name=str(tmp_path),
        num_layers=NUM_LAYERS,
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
    )


def make_nf4_state(weight_numel):
    """Create a tiny serialized NF4 state matching the fast-path format."""
    return {
        "weight": torch.arange(weight_numel, dtype=torch.uint8),
        "weight.absmax": torch.arange(4, dtype=torch.float32),
        "weight.quant_map": torch.linspace(
            -1.0, 1.0, 16, dtype=torch.float32
        ),
        "weight.quant_state.bitsandbytes__nf4": torch.tensor(
            [64, 0, 0, 0, 0, 0, 0, 0],
            dtype=torch.uint8,
        ),
    }


def make_prequantized_expert_store(tmp_path, layer_id=0, expert_id=0):
    store = tmp_path / "experts"
    store.mkdir()

    states = {
        "w1": make_nf4_state(64),
        "w2": make_nf4_state(128),
        "w3": make_nf4_state(64),
    }

    path = store / f"layer_{layer_id:02d}_expert_{expert_id:02d}.safetensors"

    tensors = {}
    for projection, state in states.items():
        for key, tensor in state.items():
            tensors[f"{projection}.{key}"] = tensor

    save_file(tensors, str(path))

    return store, states, path


def assert_state_equal(actual, expected):
    assert actual.keys() == expected.keys()

    for key in expected:
        assert torch.equal(actual[key], expected[key]), (
            f"Mismatch for {key!r}"
        )


def test_load_embeddings(tmp_path):
    tensors = make_checkpoint(tmp_path)
    loader = make_loader(tmp_path)

    loader.load()

    result = loader.load_embeddings()

    assert torch.equal(
        result,
        tensors["model.embed_tokens.weight"],
    )
    assert result.shape == (VOCAB, HIDDEN)
    assert result.dtype == torch.bfloat16
    assert result.device.type == "cpu"


def test_load_layer(tmp_path):
    tensors = make_checkpoint(tmp_path)
    loader = make_loader(tmp_path)

    loader.load()

    layer = loader.load_layer(0)

    assert isinstance(layer, LayerWeights)

    assert torch.equal(
        layer.input_layernorm,
        tensors["model.layers.0.input_layernorm.weight"],
    )

    assert torch.equal(
        layer.q_proj,
        tensors["model.layers.0.self_attn.q_proj.weight"],
    )

    assert torch.equal(
        layer.k_proj,
        tensors["model.layers.0.self_attn.k_proj.weight"],
    )

    assert torch.equal(
        layer.v_proj,
        tensors["model.layers.0.self_attn.v_proj.weight"],
    )

    assert torch.equal(
        layer.o_proj,
        tensors["model.layers.0.self_attn.o_proj.weight"],
    )

    assert torch.equal(
        layer.post_attention_layernorm,
        tensors[
            "model.layers.0.post_attention_layernorm.weight"
        ],
    )

    assert torch.equal(
        layer.moe_gate,
        tensors[
            "model.layers.0.block_sparse_moe.gate.weight"
        ],
    )


def test_load_final_norm(tmp_path):
    tensors = make_checkpoint(tmp_path)
    loader = make_loader(tmp_path)

    loader.load()

    result = loader.load_final_norm()

    assert torch.equal(
        result,
        tensors["model.norm.weight"],
    )
    assert result.shape == (HIDDEN,)
    assert result.dtype == torch.bfloat16
    assert result.device.type == "cpu"


def test_load_lm_head(tmp_path):
    tensors = make_checkpoint(tmp_path)
    loader = make_loader(tmp_path)

    loader.load()

    result = loader.load_lm_head()

    assert torch.equal(
        result,
        tensors["lm_head.weight"],
    )
    assert result.shape == (VOCAB, HIDDEN)
    assert result.dtype == torch.bfloat16
    assert result.device.type == "cpu"


def test_load_layer_rejects_invalid_layer(tmp_path):
    make_checkpoint(tmp_path)
    loader = make_loader(tmp_path)

    loader.load()

    try:
        loader.load_layer(NUM_LAYERS)
    except IndexError:
        pass
    else:
        raise AssertionError("Expected IndexError")


def test_resolve_expert_path_finds_expert(tmp_path):
    store1 = tmp_path / "store1"
    store2 = tmp_path / "store2"
    store1.mkdir()
    store2.mkdir()

    expected = store2 / "layer_26_expert_03.safetensors"
    expected.touch()

    loader = ModelLoader(
        model_name="test-model",
        num_layers=32,
        num_experts=8,
        hidden_size=4096,
        intermediate_size=14336,
        expert_store_paths=[store1, store2],
    )

    assert loader._resolve_expert_path(26, 3) == expected


def test_resolve_expert_path_raises_when_missing(tmp_path):
    loader = ModelLoader(
        model_name="test-model",
        num_layers=32,
        num_experts=8,
        hidden_size=4096,
        intermediate_size=14336,
        expert_store_paths=[tmp_path],
    )

    with pytest.raises(
        FileNotFoundError,
        match="layer_26_expert_03.safetensors",
    ):
        loader._resolve_expert_path(26, 3)


def test_load_expert_returns_prequantized_fast_path(tmp_path):
    make_checkpoint(tmp_path)
    store, states, _ = make_prequantized_expert_store(tmp_path)

    loader = ModelLoader(
        model_name=str(tmp_path),
        num_layers=NUM_LAYERS,
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        expert_store_paths=[store],
    )
    loader.load()

    expert = loader.load_expert(0, 0)

    assert isinstance(expert, PrequantizedNF4Expert)
    assert expert.device.type == "cpu"

    assert_state_equal(expert.w1, states["w1"])
    assert_state_equal(expert.w2, states["w2"])
    assert_state_equal(expert.w3, states["w3"])


def test_load_expert_does_not_use_old_prequantized_constructor(
    tmp_path,
    monkeypatch,
):
    make_checkpoint(tmp_path)
    store, _, _ = make_prequantized_expert_store(tmp_path)

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "QuantizedMixtralExpert.from_prequantized() must not be "
            "used by the expert-store fast path"
        )

    monkeypatch.setattr(
        QuantizedMixtralExpert,
        "from_prequantized",
        fail_if_called,
    )

    loader = ModelLoader(
        model_name=str(tmp_path),
        num_layers=NUM_LAYERS,
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        expert_store_paths=[store],
    )
    loader.load()

    expert = loader.load_expert(0, 0)

    assert isinstance(expert, PrequantizedNF4Expert)


def test_load_expert_uses_second_store_when_first_does_not_contain_it(
    tmp_path,
):
    make_checkpoint(tmp_path)

    store1 = tmp_path / "store1"
    store2 = tmp_path / "store2"
    store1.mkdir()
    store2.mkdir()

    states = {
        "w1": make_nf4_state(64),
        "w2": make_nf4_state(128),
        "w3": make_nf4_state(64),
    }

    path = store2 / "layer_01_expert_01.safetensors"

    tensors = {
        f"{projection}.{key}": tensor
        for projection, state in states.items()
        for key, tensor in state.items()
    }

    save_file(tensors, str(path))

    loader = ModelLoader(
        model_name=str(tmp_path),
        num_layers=NUM_LAYERS,
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        expert_store_paths=[store1, store2],
    )
    loader.load()

    expert = loader.load_expert(1, 1)

    assert isinstance(expert, PrequantizedNF4Expert)
    assert_state_equal(expert.w1, states["w1"])
    assert_state_equal(expert.w2, states["w2"])
    assert_state_equal(expert.w3, states["w3"])
