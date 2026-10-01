from __future__ import annotations

import pytest
import torch

from model.expert import PrequantizedNF4Expert, QuantizedMixtralExpert
from transfer.nf4_reconstruct import ReconstructedNF4Expert
from transfer.nf4_transfer import transfer_staged_expert_to_gpu
from transfer.pinned_memory import PinnedMemoryBudget
from transfer.reusable_staging import ReusablePinnedStagingPool


CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device required for the prequantized fast-path integration test",
)

HIDDEN = 64
INTERMEDIATE = 128


def _extract_projection_state(
    state_dict: dict[str, torch.Tensor],
    projection: str,
) -> dict[str, torch.Tensor]:
    prefix = f"{projection}."
    result = {
        key[len(prefix):]: value.detach().cpu().clone()
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }

    assert "weight" in result
    assert "weight.absmax" in result
    assert "weight.quant_map" in result
    assert "weight.quant_state.bitsandbytes__nf4" in result

    return result


def _make_real_nf4_pair():
    """Create a real bnb NF4 expert and a lightweight persisted-state expert."""

    w1 = torch.randn(
        INTERMEDIATE,
        HIDDEN,
        dtype=torch.bfloat16,
        device="cpu",
    )
    w2 = torch.randn(
        HIDDEN,
        INTERMEDIATE,
        dtype=torch.bfloat16,
        device="cpu",
    )
    w3 = torch.randn(
        INTERMEDIATE,
        HIDDEN,
        dtype=torch.bfloat16,
        device="cpu",
    )

    # This is only test-fixture construction: we use bnb once to produce
    # genuine serialized NF4 tensors. The fast path under test starts after
    # this point and must not reconstruct through from_prequantized().
    normal_gpu_expert = QuantizedMixtralExpert(w1, w2, w3).cuda()
    torch.cuda.synchronize()

    state_dict = {
        key: value.detach().cpu().clone()
        for key, value in normal_gpu_expert.state_dict().items()
    }

    prequantized_expert = PrequantizedNF4Expert.from_serialized(
        w1=_extract_projection_state(state_dict, "w1"),
        w2=_extract_projection_state(state_dict, "w2"),
        w3=_extract_projection_state(state_dict, "w3"),
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
    )

    return normal_gpu_expert, prequantized_expert


@CUDA_REQUIRED
def test_prequantized_nf4_full_cuda_fast_path():
    """
    Validate the complete one-expert fast path:

        persisted NF4 state
        -> PrequantizedNF4Expert
        -> reusable pinned staging
        -> H2D transfer
        -> direct GPU reconstruction
        -> forward

    The reconstructed expert must match a normal bitsandbytes NF4 expert
    built from the same underlying packed state.
    """
    normal_gpu_expert, prequantized_expert = _make_real_nf4_pair()

    budget = PinnedMemoryBudget(
        budget_bytes=64 * 1024 * 1024,
    )
    staging_pool = ReusablePinnedStagingPool(
        budget=budget,
        slot_size_bytes=16 * 1024 * 1024,
    )

    transfer_stream = torch.cuda.Stream()

    slot = staging_pool.acquire()
    assert slot is not None

    staging_pool.stage_expert(
        slot,
        prequantized_expert,
        layer_id=0,
        expert_id=0,
    )

    assert slot.expert_key == (0, 0)
    assert staging_pool.active_slots == 1

    gpu_state = transfer_staged_expert_to_gpu(
        slot,
        transfer_stream,
    )

    transfer_stream.synchronize()

    reconstructed = ReconstructedNF4Expert(
        prequantized_expert,
        gpu_state,
    )

    torch.cuda.synchronize()

    x = torch.randn(
        2,
        HIDDEN,
        device="cuda",
        dtype=torch.float16,
    )

    expected = normal_gpu_expert(x)
    actual = reconstructed(x)

    torch.cuda.synchronize()

    assert actual.shape == expected.shape == (2, HIDDEN)
    assert actual.device.type == "cuda"
    assert actual.dtype == torch.float16

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-3,
        atol=1e-3,
    )

    # The staging slot may now be released because reconstruction owns
    # independent GPU tensors.
    staging_pool.mark_transfer_complete(
        slot,
        transfer_stream,
    )
    staging_pool.release(slot)

    assert staging_pool.active_slots == 0


@CUDA_REQUIRED
def test_prequantized_fast_path_reuses_staging_slot():
    """A completed transfer slot can be reused for another expert."""

    _, prequantized_expert = _make_real_nf4_pair()

    budget = PinnedMemoryBudget(
        budget_bytes=64 * 1024 * 1024,
    )
    staging_pool = ReusablePinnedStagingPool(
        budget=budget,
        slot_size_bytes=16 * 1024 * 1024,
    )

    stream = torch.cuda.Stream()

    first = staging_pool.acquire()
    assert first is not None

    staging_pool.stage_expert(
        first,
        prequantized_expert,
        layer_id=0,
        expert_id=0,
    )

    transfer_staged_expert_to_gpu(first, stream)
    stream.synchronize()

    staging_pool.mark_transfer_complete(first, stream)
    staging_pool.release(first)

    assert staging_pool.active_slots == 0
    assert staging_pool.slot_count == 1

    second = staging_pool.acquire()
    assert second is first
    assert staging_pool.active_slots == 1

    staging_pool.release(second)
