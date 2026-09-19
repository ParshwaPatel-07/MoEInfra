import os
from pathlib import Path

import torch
import bitsandbytes as bnb
from safetensors import safe_open


MODEL_PATH = Path(
    "/kaggle/input/models/mistral-ai/mixtral/"
    "pytorch/8x7b-instruct-v0.1-hf/1"
)

LAYER_ID = 0
EXPERT_ID = 0


def find_expert_tensors(layer_id: int, expert_id: int):
    """Find the shard containing one expert's w1/w2/w3 tensors."""

    index_path = MODEL_PATH / "model.safetensors.index.json"

    import json

    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    prefix = (
        f"model.layers.{layer_id}."
        f"block_sparse_moe.experts.{expert_id}."
    )

    names = {
        "w1": prefix + "w1.weight",
        "w2": prefix + "w2.weight",
        "w3": prefix + "w3.weight",
    }

    shards = {weight_map[name] for name in names.values()}

    if len(shards) != 1:
        raise RuntimeError(
            f"Expert tensors are spread across multiple shards: {shards}"
        )

    shard = MODEL_PATH / next(iter(shards))

    return names, shard


def quantize_linear(weight: torch.Tensor) -> bnb.nn.Linear4bit:
    """Create one NF4 Linear4bit from a BF16 weight tensor."""

    linear = bnb.nn.Linear4bit(
        input_features=weight.shape[1],
        output_features=weight.shape[0],
        bias=False,
        quant_type="nf4",
        compress_statistics=False,
    )

    linear.weight = bnb.nn.Params4bit(
        weight,
        requires_grad=False,
        quant_type="nf4",
    )

    return linear


def inspect_linear(name: str, linear: bnb.nn.Linear4bit):
    print(f"\n{name}")
    print(f"  weight device: {linear.weight.device}")
    print(f"  weight dtype:  {linear.weight.dtype}")

    quant_state = linear.weight.quant_state

    print(f"  quant_state:   {quant_state}")

    if quant_state is not None:
        print(f"  blocksize:     {quant_state.blocksize}")
        print(f"  quant_type:    {quant_state.quant_type}")

        if hasattr(quant_state, "absmax"):
            print(f"  absmax device: {quant_state.absmax.device}")
            print(f"  absmax dtype:  {quant_state.absmax.dtype}")


def main():
    print("=" * 70)
    print("MoEInfra Lab — Single Expert NF4 Experiment")
    print("=" * 70)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model path does not exist:\n{MODEL_PATH}"
        )

    print(f"\nModel path: {MODEL_PATH}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a CUDA GPU.")

    print(f"GPU: {torch.cuda.get_device_name()}")

    # ------------------------------------------------------------
    # 1. Find the expert tensors
    # ------------------------------------------------------------

    names, shard = find_expert_tensors(LAYER_ID, EXPERT_ID)

    print(f"\nLayer: {LAYER_ID}")
    print(f"Expert: {EXPERT_ID}")
    print(f"Shard: {shard.name}")

    # ------------------------------------------------------------
    # 2. Load the original BF16 tensors
    # ------------------------------------------------------------

    weights = {}

    with safe_open(shard, framework="pt", device="cpu") as f:
        for name, tensor_name in names.items():
            tensor = f.get_tensor(tensor_name)
            weights[name] = tensor

            print(f"\n{name}")
            print(f"  shape: {tuple(tensor.shape)}")
            print(f"  dtype: {tensor.dtype}")
            print(
                f"  size: "
                f"{tensor.numel() * tensor.element_size() / 1024**2:.2f} MiB"
            )

    # ------------------------------------------------------------
    # 3. Quantize each matrix to NF4
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("Creating NF4 layers")
    print("=" * 70)

    layers = {}

    for name, weight in weights.items():
        layers[name] = quantize_linear(weight)

        print(
            f"{name}: created "
            f"{weight.numel() * weight.element_size() / 1024**2:.2f} MiB "
            "BF16 source"
        )

    # ------------------------------------------------------------
    # 4. Move to GPU — this triggers bnb quantization
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("Moving NF4 layers to GPU")
    print("=" * 70)

    for name, layer in layers.items():
        layer = layer.cuda()
        layers[name] = layer

        inspect_linear(name, layer)

    torch.cuda.synchronize()

    # ------------------------------------------------------------
    # 5. GPU memory usage
    # ------------------------------------------------------------

    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2

    print("\n" + "=" * 70)
    print("GPU memory")
    print("=" * 70)

    print(f"Allocated: {allocated:.2f} MiB")
    print(f"Reserved:  {reserved:.2f} MiB")

    # ------------------------------------------------------------
    # 6. Simple forward test
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("Forward test")
    print("=" * 70)

    x = torch.randn(
        1,
        weights["w1"].shape[1],
        device="cuda",
        dtype=torch.float16,
    )

    with torch.no_grad():
        y1 = layers["w1"](x)
        y3 = layers["w3"](x)
        y = layers["w2"](torch.nn.functional.silu(y1) * y3)

    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(y.shape)}")
    print(f"Output dtype: {y.dtype}")
    print(f"Output device: {y.device}")

    # ------------------------------------------------------------
    # 7. Move back to CPU and inspect quant_state
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("GPU → CPU test")
    print("=" * 70)

    for name, layer in layers.items():
        layer = layer.cpu()
        layers[name] = layer

        inspect_linear(name, layer)

    print("\nExperiment completed.")
    print("No full Mixtral model was loaded.")


if __name__ == "__main__":
    main()