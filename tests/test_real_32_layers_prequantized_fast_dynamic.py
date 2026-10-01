from __future__ import annotations

import gc
import os
import time

import pytest
import torch
from transformers import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralRotaryEmbedding

from cache.manager import CacheManager
from cache.types import EvictionPolicy
from engine.decoder_layer import MoEInfraDecoderLayer
from engine.router import ExpertRouter
from model.loader import ModelLoader
from model.moe_layer import MoELayer
from transfer.pinned_memory import PinnedMemoryBudget
from transfer.reusable_staging import ReusablePinnedStagingPool
from transfer.scheduler import TransferScheduler
import transfer.scheduler as scheduler_module


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
def test_real_mixtral_32_layer_chain_on_t4_prequantized_fast_path():
    """System-scale validation of the persisted-NF4 fast path."""

    timing = {
        "load_expert_ms": 0.0,
        "staging_ms": 0.0,
        "h2d_ms": 0.0,
        "reconstruction_ms": 0.0,
        "decoder_ms": 0.0,
        "load_expert_calls": 0,
        "h2d_bytes": 0,
    }

    original_load_expert = ModelLoader.load_expert
    original_stage_expert = ReusablePinnedStagingPool.stage_expert
    original_transfer = scheduler_module.transfer_staged_expert_to_gpu
    original_reconstruct = scheduler_module.ReconstructedNF4Expert

    def timed_load_expert(self, layer_id, expert_id):
        start = time.perf_counter()
        result = original_load_expert(self, layer_id, expert_id)
        timing["load_expert_ms"] += (time.perf_counter() - start) * 1000.0
        timing["load_expert_calls"] += 1
        return result

    def timed_stage_expert(self, slot, expert, layer_id, expert_id):
        start = time.perf_counter()
        result = original_stage_expert(
            self, slot, expert, layer_id, expert_id
        )
        timing["staging_ms"] += (time.perf_counter() - start) * 1000.0
        return result

    def timed_transfer(slot, stream):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        with torch.cuda.stream(stream):
            start.record(stream)
            gpu_state = original_transfer(slot, stream)
            end.record(stream)

        end.synchronize()
        timing["h2d_ms"] += start.elapsed_time(end)

        for state in gpu_state.values():
            for tensor in state.values():
                if tensor is not None:
                    timing["h2d_bytes"] += tensor.numel() * tensor.element_size()

        return gpu_state

    def timed_reconstruct(*args, **kwargs):
        start = time.perf_counter()
        result = original_reconstruct(*args, **kwargs)
        timing["reconstruction_ms"] += (
            time.perf_counter() - start
        ) * 1000.0
        return result

    ModelLoader.load_expert = timed_load_expert
    ReusablePinnedStagingPool.stage_expert = timed_stage_expert
    scheduler_module.transfer_staged_expert_to_gpu = timed_transfer
    scheduler_module.ReconstructedNF4Expert = timed_reconstruct

    try:
        loader = ModelLoader(
            model_name=MODEL_PATH,
            num_layers=32,
            num_experts=8,
            hidden_size=4096,
            intermediate_size=14336,
            expert_store_paths=[STORE1, STORE2],
        )
        loader.load()

        config = MixtralConfig.from_pretrained(MODEL_PATH)
        config._attn_implementation = "eager"

        assert config.hidden_size == 4096
        assert config.num_hidden_layers == 32
        assert config.num_local_experts == 8
        assert config.num_experts_per_tok == 2

        profile_layers = int(
            os.environ.get("MOEINFRA_PROFILE_LAYERS", str(config.num_hidden_layers))
        )
        if not 1 <= profile_layers <= config.num_hidden_layers:
            raise ValueError(
                f"MOEINFRA_PROFILE_LAYERS must be between 1 and "
                f"{config.num_hidden_layers}, got {profile_layers}"
            )

        cache_manager = CacheManager(
            gpu_slots=1,
            cpu_slots=8,
            policy=EvictionPolicy.LRU,
        )

        staging_pool = ReusablePinnedStagingPool(
            budget=PinnedMemoryBudget(512 * 1024 * 1024),
            slot_size_bytes=128 * 1024 * 1024,
        )

        transfer_stream = torch.cuda.Stream()

        transfer_scheduler = TransferScheduler(
            cache_manager=cache_manager,
            bandwidth_gbps=10.0,
            max_concurrent=1,
            staging_pool=staging_pool,
            transfer_stream=transfer_stream,
        )

        batch_size = 1
        sequence_length = 4
        hidden_size = config.hidden_size

        hidden_states = torch.randn(
            batch_size,
            sequence_length,
            hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        initial_hidden_states = hidden_states.clone()

        position_ids = torch.arange(
            sequence_length, device="cuda"
        ).unsqueeze(0)
        cache_position = torch.arange(
            sequence_length, device="cuda"
        )

        rotary_emb = MixtralRotaryEmbedding(config)
        position_embeddings = rotary_emb(
            hidden_states,
            position_ids=position_ids,
        )

        total_start = time.perf_counter()
        layer_outputs = []

        for layer_id in range(profile_layers):
            print(f"\n===== Layer {layer_id} =====")

            weights = loader.load_layer(layer_id)

            decoder_layer = MoEInfraDecoderLayer(
                config,
                layer_idx=layer_id,
            ).to(device="cuda", dtype=torch.bfloat16)

            with torch.no_grad():
                decoder_layer.input_layernorm.weight.copy_(
                    weights.input_layernorm.cuda()
                )
                decoder_layer.self_attn.q_proj.weight.copy_(
                    weights.q_proj.cuda()
                )
                decoder_layer.self_attn.k_proj.weight.copy_(
                    weights.k_proj.cuda()
                )
                decoder_layer.self_attn.v_proj.weight.copy_(
                    weights.v_proj.cuda()
                )
                decoder_layer.self_attn.o_proj.weight.copy_(
                    weights.o_proj.cuda()
                )
                decoder_layer.post_attention_layernorm.weight.copy_(
                    weights.post_attention_layernorm.cuda()
                )

            router = ExpertRouter(
                gate_weight=weights.moe_gate.cuda(),
                num_experts_per_tok=config.num_experts_per_tok,
            )

            moe_layer = MoELayer(
                layer_id=layer_id,
                router=router,
                cache_manager=cache_manager,
                model_loader=loader,
                transfer_scheduler=transfer_scheduler,
            )
            decoder_layer.set_moe(moe_layer)

            start = time.perf_counter()
            with torch.no_grad():
                output = decoder_layer(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=None,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    past_key_values=None,
                )
            torch.cuda.synchronize()
            timing["decoder_ms"] += (time.perf_counter() - start) * 1000.0

            assert output.shape == (
                batch_size, sequence_length, hidden_size
            )
            assert output.device.type == "cuda"
            assert output.dtype == torch.bfloat16
            assert torch.isfinite(output).all()

            stats = cache_manager.stats()
            assert stats.gpu_slots_used <= 1

            print(
                f"output shape: {tuple(output.shape)}, "
                f"cache misses: {stats.misses}, hits: {stats.hits}, "
                f"GPU slots: {stats.gpu_slots_used}"
            )

            hidden_states = output
            layer_outputs.append(output)

            del decoder_layer, moe_layer, router, weights
            gc.collect()
            torch.cuda.empty_cache()

        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - total_start) * 1000.0

        stats = cache_manager.stats()

        assert len(layer_outputs) == profile_layers
        assert hidden_states.shape == (
            batch_size, sequence_length, hidden_size
        )
        assert hidden_states.device.type == "cuda"
        assert hidden_states.dtype == torch.bfloat16
        assert torch.isfinite(hidden_states).all()
        assert not torch.equal(initial_hidden_states, hidden_states)
        assert stats.gpu_slots_used <= 1
        assert stats.misses > 0
        assert timing["load_expert_calls"] == stats.misses

        print("\n========================================")
        print(f"{profile_layers}-LAYER PREQUANTIZED FAST-PATH TEST PASSED")
        print("========================================")
        print(f"Total wall time:       {total_ms:.2f} ms")
        print(f"load_expert total:     {timing['load_expert_ms']:.2f} ms")
        print(f"staging/pinning:       {timing['staging_ms']:.2f} ms")
        print(f"H2D transfer:          {timing['h2d_ms']:.2f} ms")
        print(f"NF4 reconstruction:    {timing['reconstruction_ms']:.2f} ms")
        print(f"decoder compute:       {timing['decoder_ms']:.2f} ms")
        print(f"load_expert calls:     {timing['load_expert_calls']}")
        print(
            f"H2D bytes measured:    "
            f"{timing['h2d_bytes'] / 1024**3:.3f} GiB"
        )
        print(f"Total cache misses:    {stats.misses}")
        print(f"Total cache hits:      {stats.hits}")
        print(f"GPU slots used:        {stats.gpu_slots_used}")
        print(f"CPU slots used:        {stats.cpu_slots_used}")
        print(f"Total evictions:       {stats.evictions}")

    finally:
        ModelLoader.load_expert = original_load_expert
        ReusablePinnedStagingPool.stage_expert = original_stage_expert
        scheduler_module.transfer_staged_expert_to_gpu = original_transfer
        scheduler_module.ReconstructedNF4Expert = original_reconstruct
        torch.cuda.synchronize()
