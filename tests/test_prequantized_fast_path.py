from __future__ import annotations

import torch
import pytest
import bitsandbytes as bnb

from model.expert import PrequantizedNF4Expert


HIDDEN_SIZE = 8
INTERMEDIATE_SIZE = 16


def make_nf4_state(
    weight_numel: int,
) -> dict[str, torch.Tensor]:
    """Create a small serialized NF4 state for unit testing.

    The values are arbitrary; this test checks representation preservation,
    not NF4 numerical reconstruction.
    """
    return {
        "weight": torch.arange(weight_numel, dtype=torch.uint8),
        "weight.absmax": torch.arange(4, dtype=torch.float32),
        "weight.quant_map": torch.linspace(-1.0, 1.0, 16, dtype=torch.float32),
        "weight.quant_state.bitsandbytes__nf4": torch.tensor(
            [64, 0, 0, 0, 0, 0, 0, 0],
            dtype=torch.uint8,
        ),
    }


def assert_state_equal(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> None:
    assert actual.keys() == expected.keys()

    for key in expected:
        assert torch.equal(actual[key], expected[key]), f"Mismatch for {key!r}"


def make_expert() -> tuple[
    PrequantizedNF4Expert,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    w1 = make_nf4_state(64)
    w2 = make_nf4_state(128)
    w3 = make_nf4_state(64)

    expert = PrequantizedNF4Expert.from_serialized(
        w1=w1,
        w2=w2,
        w3=w3,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
    )

    return expert, w1, w2, w3


def test_from_serialized_preserves_all_nf4_state() -> None:
    expert, w1, w2, w3 = make_expert()

    assert_state_equal(expert.w1, w1)
    assert_state_equal(expert.w2, w2)
    assert_state_equal(expert.w3, w3)

    assert expert.hidden_size == HIDDEN_SIZE
    assert expert.intermediate_size == INTERMEDIATE_SIZE


def test_from_serialized_does_not_construct_bitsandbytes_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CPU fast path must not use the slow bnb prequantized constructor."""

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "Params4bit.from_prequantized() must not be called by "
            "PrequantizedNF4Expert.from_serialized()"
        )

    monkeypatch.setattr(
        bnb.nn.Params4bit,
        "from_prequantized",
        fail_if_called,
    )

    expert, _, _, _ = make_expert()

    assert isinstance(expert, PrequantizedNF4Expert)


def test_prequantized_expert_reports_cpu_device() -> None:
    expert, _, _, _ = make_expert()

    assert expert.device.type == "cpu"


def test_prequantized_expert_has_nonzero_size() -> None:
    expert, _, _, _ = make_expert()

    assert expert.size_bytes > 0


@pytest.mark.parametrize(
    "projection",
    ["w1", "w2", "w3"],
)
def test_missing_required_tensor_is_rejected(projection: str) -> None:
    w1 = make_nf4_state(64)
    w2 = make_nf4_state(128)
    w3 = make_nf4_state(64)

    del {"w1": w1, "w2": w2, "w3": w3}[projection]["weight.quant_map"]

    with pytest.raises((ValueError, KeyError), match="quant_map|NF4|state"):
        PrequantizedNF4Expert.from_serialized(
            w1=w1,
            w2=w2,
            w3=w3,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
        )
