import json

import torch
from safetensors.torch import save_file

from model.loader import ModelLoader
from model.types import LayerWeights
import pytest
from model.expert import QuantizedMixtralExpert

HIDDEN = 8
INTERMEDIATE = 16
VOCAB = 12
NUM_LAYERS = 2
NUM_EXPERTS = 2

def make_prequantized_expert_store(tmp_path):
    store = tmp_path / "expert_store"
    store.mkdir()

    torch.manual_seed(0)

    expert = QuantizedMixtralExpert(
        torch.randn(INTERMEDIATE, HIDDEN, dtype=torch.bfloat16),
        torch.randn(HIDDEN, INTERMEDIATE, dtype=torch.bfloat16),
        torch.randn(INTERMEDIATE, HIDDEN, dtype=torch.bfloat16),
    )

    if not torch.cuda.is_available():
        pytest.skip("CUDA required to create a real NF4 test fixture")

    expert = expert.cuda()
    torch.cuda.synchronize()

    state = {}

    for name, tensor in expert.state_dict().items():
        state[name] = tensor.cpu()

    save_file(
        state,
        str(store / "layer_00_expert_00.safetensors"),
    )

    return store

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

    with pytest.raises(FileNotFoundError, match="layer_26_expert_03.safetensors"):
        loader._resolve_expert_path(26, 3)

def test_load_expert_from_prequantized_store_stays_on_cpu(tmp_path):
    make_checkpoint(tmp_path)
    store = make_prequantized_expert_store(tmp_path)

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

    assert expert.device.type == "cpu"
    assert expert.w1.weight.device.type == "cpu"
    assert expert.w2.weight.device.type == "cpu"
    assert expert.w3.weight.device.type == "cpu"

def test_prequantized_expert_cpu_to_gpu_forward(tmp_path):
    make_checkpoint(tmp_path)
    store = make_prequantized_expert_store(tmp_path)

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

    assert expert.device.type == "cpu"

    x = torch.randn(2, HIDDEN, dtype=torch.float16)

    gpu_expert = expert.cuda()
    y = gpu_expert(x.cuda())

    assert gpu_expert.device.type == "cuda"
    assert y.device.type == "cuda"
    assert y.shape == (2, HIDDEN)
    assert torch.isfinite(y).all()

def test_real_prequantized_expert_stores():
    store1 = "/kaggle/input/datasets/parshwapatel07/mixtral-8x7b-nf4-experts/experts"
    store2 = "/kaggle/input/datasets/parshwapatel07/mixtral-8x7b-nf4-experts-part2/experts"

    loader = ModelLoader(
        model_name="/kaggle/input/models/mistral-ai/mixtral/pytorch/8x7b-instruct-v0.1-hf/1",
        num_layers=32,
        num_experts=8,
        hidden_size=4096,
        intermediate_size=14336,
        expert_store_paths=[store1, store2],
    )

    loader.load()

    expert_a = loader.load_expert(0, 0)
    expert_b = loader.load_expert(26, 3)

    assert expert_a.device.type == "cpu"
    assert expert_b.device.type == "cpu"

    assert expert_a.w1.weight.device.type == "cpu"
    assert expert_b.w1.weight.device.type == "cpu"

    print("L0 E0:", expert_a.size_bytes / 1024**2, "MiB")
    print("L26 E3:", expert_b.size_bytes / 1024**2, "MiB")