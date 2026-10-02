from __future__ import annotations

import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.nn.functional as F


_NF4_BLOCKSIZE = 64
_NF4_QUANT_TYPE = "nf4"
_NF4_QUANT_STORAGE = torch.uint8
_NF4_DTYPE = torch.bfloat16
_NESTED_BLOCKSIZE = 256
_NESTED_DTYPE = torch.float32


def _build_quant_state(
    state: dict[str, torch.Tensor | None],
    shape: tuple[int, int],
) -> bnb.functional.QuantState:
    """Build a bitsandbytes QuantState directly from transferred tensors."""
    state2 = None

    state2_absmax = state.get("state2_absmax")
    state2_code = state.get("state2_code")
    state2_offset = state.get("state2_offset")

    if state2_absmax is not None or state2_code is not None:
        if state2_absmax is None or state2_code is None:
            raise ValueError("Incomplete nested NF4 quantization state")

        # The nested state quantizes the main `absmax` tensor, so its shape is
        # the shape of the main state's absmax tensor, not the shape of the
        # nested absmax tensor itself.
        state2 = bnb.functional.QuantState(
            absmax=state2_absmax,
            shape=state["absmax"].shape,
            code=state2_code,
            blocksize=_NESTED_BLOCKSIZE,
            quant_type=None,
            dtype=_NESTED_DTYPE,
            offset=state2_offset,
        )

    return bnb.functional.QuantState(
        absmax=state["absmax"],
        shape=torch.Size(shape),
        code=state["code"],
        blocksize=_NF4_BLOCKSIZE,
        quant_type=_NF4_QUANT_TYPE,
        dtype=_NF4_DTYPE,
        offset=state.get("offset"),
        state2=state2,
    )


def _reconstruct_linear(
    state: dict[str, torch.Tensor | None],
    *,
    in_features: int,
    out_features: int,
) -> bnb.nn.Linear4bit:
    """Create a GPU Linear4bit shell around already-transferred NF4 data.

    This deliberately avoids both:
      * Params4bit.from_prequantized()
      * the normal Linear4bit constructor

    The first would decode/rebuild quantization metadata through the slow
    convenience path; the second performs an unnecessary parameter
    initialization. The packed NF4 tensors and QuantState are already known.
    """
    weight = state["weight"]
    if weight is None:
        raise ValueError("NF4 state is missing packed weight")

    if weight.dtype != _NF4_QUANT_STORAGE:
        raise ValueError(
            f"Expected packed NF4 weight dtype {_NF4_QUANT_STORAGE}, "
            f"got {weight.dtype}"
        )

    shape = (out_features, in_features)
    quant_state = _build_quant_state(state, shape)

    # Bypass Linear4bit.__init__ so we do not allocate and initialize a large
    # floating-point placeholder tensor.
    linear = object.__new__(bnb.nn.Linear4bit)
    nn.Module.__init__(linear)

    linear.in_features = in_features
    linear.out_features = out_features
    linear.bias = None
    linear.compute_dtype = torch.float16
    linear.compute_type_is_set = True
    linear.quant_storage = _NF4_QUANT_STORAGE
    linear.support_avx512bf16_for_cpu = False
    linear.quant_state = quant_state

    # Params4bit construction is metadata-only: it wraps the already-packed
    # uint8 CUDA tensor without re-quantizing or allocating an FP tensor.
    linear.weight = bnb.nn.Params4bit(
        weight,
        requires_grad=False,
        quant_state=quant_state,
        blocksize=quant_state.blocksize,
        compress_statistics=quant_state.nested,
        quant_type=_NF4_QUANT_TYPE,
        quant_storage=_NF4_QUANT_STORAGE,
        module=None,
        bnb_quantized=True,
    )

    return linear


class ReconstructedNF4Expert(nn.Module):
    """GPU-resident Mixtral expert reconstructed from transferred NF4 state."""

    def __init__(self, original_expert, gpu_state) -> None:
        super().__init__()

        hidden_size = getattr(original_expert, "hidden_size", None)
        intermediate_size = getattr(original_expert, "intermediate_size", None)

        if hidden_size is None or intermediate_size is None:
            raise ValueError(
                "Prequantized expert must expose hidden_size and intermediate_size"
            )

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.w1 = _reconstruct_linear(
            gpu_state["w1"],
            in_features=hidden_size,
            out_features=intermediate_size,
        )

        self.w2 = _reconstruct_linear(
            gpu_state["w2"],
            in_features=intermediate_size,
            out_features=hidden_size,
        )

        self.w3 = _reconstruct_linear(
            gpu_state["w3"],
            in_features=hidden_size,
            out_features=intermediate_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        up = self.w3(x)
        return self.w2(F.silu(gate) * up)

    @property
    def device(self) -> torch.device:
        return self.w1.weight.device

    @property
    def size_bytes(self) -> int:
        return sum(
            layer.weight.numel() * layer.weight.element_size()
            for layer in (self.w1, self.w2, self.w3)
        )
