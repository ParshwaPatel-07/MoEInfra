import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb
from torch.nn.utils import skip_init

class QuantizedMixtralExpert(nn.Module):
    """
    A single Mixtral MoE expert using NF4 quantization.

    The expert consists of three Linear4bit layers:
        w1: hidden_size -> intermediate_size
        w3: hidden_size -> intermediate_size
        w2: intermediate_size -> hidden_size

    Quantization happens once during construction.
    """

    def __init__(
        self,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
    ) -> None:
        super().__init__()

        self.w1 = self._quantize_linear(w1)
        self.w2 = self._quantize_linear(w2)
        self.w3 = self._quantize_linear(w3)

    @staticmethod
    def _quantize_linear(weight: torch.Tensor) -> bnb.nn.Linear4bit:
        if weight.ndim != 2:
            raise ValueError(
                f"Expected a 2D weight tensor, got shape {weight.shape}"
            )

        if weight.dtype != torch.bfloat16:
            raise ValueError(
                f"Expected BF16 weights, got {weight.dtype}"
            )

        linear = object.__new__(bnb.nn.Linear4bit)
        torch.nn.Module.__init__(linear)

        linear.in_features = weight.shape[1]
        linear.out_features = weight.shape[0]
        linear.weight = bnb.nn.Params4bit(
            weight,
            requires_grad=False,
            compress_statistics=False,
            quant_type="nf4",
            module=linear,
        )
        linear.bias = None
        linear.compute_dtype = torch.float16
        linear.compute_type_is_set = True
        linear.quant_state = None
        linear.quant_storage = torch.uint8
        linear.support_avx512bf16_for_cpu = False

        return linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mixtral SwiGLU expert:

            w2(SiLU(w1(x)) * w3(x))
        """

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