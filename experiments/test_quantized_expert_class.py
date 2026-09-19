import json
from pathlib import Path

import torch
from safetensors import safe_open

from model.expert import QuantizedMixtralExpert


MODEL_PATH = Path(
    "/kaggle/input/models/mistral-ai/mixtral/"
    "pytorch/8x7b-instruct-v0.1-hf/1"
)

LAYER_ID = 0
EXPERT_ID = 0


def main():
    print("=" * 70)
    print("Testing QuantizedMixtralExpert")
    print("=" * 70)

    assert torch.cuda.is_available(), "CUDA GPU required"

    # ------------------------------------------------------------
    # 1. Find the expert tensors
    # ------------------------------------------------------------

    with open(MODEL_PATH / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    prefix = (
        f"model.layers.{LAYER_ID}."
        f"block_sparse_moe.experts.{EXPERT_ID}."
    )

    names = {
        "w1": prefix + "w1.weight",
        "w2": prefix + "w2.weight",
        "w3": prefix + "w3.weight",
    }

    shards = {weight_map[name] for name in names.values()}
    assert len(shards) == 1

    shard = MODEL_PATH / next(iter(shards))

    # ------------------------------------------------------------
    # 2. Load real BF16 weights
    # ------------------------------------------------------------

    weights = {}

    with safe_open(shard, framework="pt", device="cpu") as f:
        for name, tensor_name in names.items():
            weights[name] = f.get_tensor(tensor_name)

            print(
                f"{name}: "
                f"shape={tuple(weights[name].shape)}, "
                f"dtype={weights[name].dtype}"
            )

    # ------------------------------------------------------------
    # 3. Construct our actual expert
    # ------------------------------------------------------------

    print("\nCreating QuantizedMixtralExpert...")

    expert = QuantizedMixtralExpert(
        w1=weights["w1"],
        w2=weights["w2"],
        w3=weights["w3"],
    )

    print("Expert created successfully.")

    # ------------------------------------------------------------
    # 4. Verify NF4 configuration
    # ------------------------------------------------------------

    for name in ("w1", "w2", "w3"):
        layer = getattr(expert, name)

        print(f"\n{name}")
        print("  weight dtype:", layer.weight.dtype)
        print("  weight device:", layer.weight.device)
        print("  quant type:", layer.weight.quant_state.quant_type)
        print("  block size:", layer.weight.quant_state.blocksize)

        assert layer.weight.quant_state.quant_type == "nf4"
        assert layer.weight.quant_state.blocksize == 64

    # ------------------------------------------------------------
    # 5. Move entire expert to GPU
    # ------------------------------------------------------------

    print("\nMoving expert to GPU...")

    expert = expert.cuda()

    for name in ("w1", "w2", "w3"):
        layer = getattr(expert, name)

        assert layer.weight.device.type == "cuda"
        assert layer.weight.quant_state.absmax.device.type == "cuda"

    print("All weights + quantization state are on GPU.")

    # ------------------------------------------------------------
    # 6. Run actual expert forward
    # ------------------------------------------------------------

    print("\nRunning forward pass...")

    x = torch.randn(
        1,
        4096,
        device="cuda",
        dtype=torch.float16,
    )

    with torch.no_grad():
        output = expert(x)

    print("Input shape: ", tuple(x.shape))
    print("Output shape:", tuple(output.shape))
    print("Output dtype:", output.dtype)
    print("Output device:", output.device)

    assert output.shape == (1, 4096)
    assert output.device.type == "cuda"
    assert torch.isfinite(output).all()

    # ------------------------------------------------------------
    # 7. GPU → CPU
    # ------------------------------------------------------------

    print("\nMoving expert back to CPU...")

    expert = expert.cpu()

    for name in ("w1", "w2", "w3"):
        layer = getattr(expert, name)

        assert layer.weight.device.type == "cpu"
        assert layer.weight.quant_state.absmax.device.type == "cpu"

    print("All weights + quantization state are back on CPU.")

    print("\n" + "=" * 70)
    print("PASS — QuantizedMixtralExpert works")
    print("=" * 70)


if __name__ == "__main__":
    main()