from __future__ import annotations

from dataclasses import dataclass

import torch

from transfer.pinned_memory import PinnedMemoryBudget
import bitsandbytes.functional as bnb_functional

@dataclass
class StagingSlot:
    slot_id: int
    size_bytes: int
    tensors: dict[str, dict[str, torch.Tensor | None]]
    in_use: bool = False
    ready_event: torch.cuda.Event | None = None
    expert_key: tuple[int, int] | None = None


class ReusablePinnedStagingPool:
    """
    Bounded reusable pinned-memory slots.

    A slot remains unavailable until the CUDA event associated with
    its previous transfer has completed.

    CPU experts are expected to expose already-serialized NF4 state.
    The staging pool does not construct bitsandbytes Linear4bit modules
    and does not quantize or reconstruct experts.
    """

    _PROJECTIONS = ("w1", "w2", "w3")

    def __init__(
        self,
        budget: PinnedMemoryBudget,
        slot_size_bytes: int,
    ):
        if slot_size_bytes <= 0:
            raise ValueError("slot_size_bytes must be positive")

        self.budget = budget
        self.slot_size_bytes = slot_size_bytes
        self._slots: dict[int, StagingSlot] = {}

    @property
    def slot_count(self) -> int:
        return len(self._slots)

    @property
    def active_slots(self) -> int:
        return sum(slot.in_use for slot in self._slots.values())

    def _create_slot(self, slot_id: int) -> StagingSlot | None:
        if not self.budget.reserve(self.slot_size_bytes):
            return None

        try:
            slot = StagingSlot(
                slot_id=slot_id,
                size_bytes=self.slot_size_bytes,
                tensors={},
            )

            self._slots[slot_id] = slot
            return slot

        except Exception:
            self.budget.release(self.slot_size_bytes)
            raise

    def acquire(self) -> StagingSlot | None:
        # First try to reuse a completed slot.
        for slot in self._slots.values():
            if slot.in_use:
                continue

            if slot.ready_event is not None:
                if not slot.ready_event.query():
                    continue
                slot.ready_event = None

            slot.in_use = True
            return slot

        # No reusable slot exists.
        slot_id = len(self._slots)
        slot = self._create_slot(slot_id)

        if slot is not None:
            slot.in_use = True

        return slot

    def mark_transfer_complete(
        self,
        slot: StagingSlot,
        stream: torch.cuda.Stream,
    ) -> None:
        if not slot.in_use:
            raise RuntimeError("Cannot mark an inactive slot")

        event = torch.cuda.Event()

        with torch.cuda.stream(stream):
            event.record(stream)

        slot.ready_event = event

    def release(self, slot: StagingSlot) -> None:
        if not slot.in_use:
            raise RuntimeError("Slot is not in use")

        slot.in_use = False

    def destroy(self) -> None:
        self._slots.clear()
        self.budget.used_bytes = 0

    @staticmethod
    def _tensor_bytes(tensor: torch.Tensor | None) -> int:
        if tensor is None:
            return 0
        return tensor.numel() * tensor.element_size()

    @classmethod
    def _projection_state(cls, expert, name: str) -> dict[str, torch.Tensor | None]:
        """Return one projection in the canonical staging representation.

        Canonical keys are:
            weight, absmax, code, offset,
            state2_absmax, state2_code, state2_offset (optional)

        The CPU cache stores the serialized NF4 representation. We accept both
        a canonical state and the bitsandbytes Linear4bit.state_dict() form.
        """
        projection = getattr(expert, name)

        # Preferred representation: the lightweight pre-quantized expert
        # stores the serialized state dictionary directly.
        if isinstance(projection, dict):
            raw_state = projection
        else:
            raw_state = getattr(projection, "state", None)

        if not isinstance(raw_state, dict):
            raise TypeError(
                f"Expert projection {name!r} does not expose serialized NF4 state"
            )

        if "weight" not in raw_state:
            raise ValueError(f"Projection {name!r} is missing 'weight'")

        # Already canonical. Some PrequantizedNF4Expert implementations may
        # store `quant_map` instead of the canonical `code` name.
        if "absmax" in raw_state and ("code" in raw_state or "quant_map" in raw_state):
            canonical: dict[str, torch.Tensor | None] = {
                "weight": raw_state["weight"],
                "absmax": raw_state["absmax"],
                "code": raw_state.get("code", raw_state.get("quant_map")),
                "offset": raw_state.get("offset"),
            }

            for key in (
                "state2_absmax",
                "state2_code",
                "state2_offset",
            ):
                if key in raw_state:
                    canonical[key] = raw_state[key]

            if canonical["code"] is None:
                raise ValueError(f"Projection {name!r} is missing NF4 code/quant_map")

            return canonical

        # Serialized bitsandbytes state_dict representation:
        #   weight
        #   weight.absmax
        #   weight.quant_map
        #   weight.nested_absmax / weight.nested_quant_map (when nested)
        #   weight.quant_state.bitsandbytes__nf4
        required = {
            "weight.absmax",
            "weight.quant_map",
            "weight.quant_state.bitsandbytes__nf4",
        }
        missing = required - raw_state.keys()
        if missing:
            raise ValueError(
                f"Projection {name!r} is missing serialized NF4 state tensors: "
                f"{sorted(missing)}"
            )

        # QuantState.from_dict is only used to decode metadata into tensor
        # references. It does not invoke Params4bit.from_prequantized() and
        # does not construct a Linear4bit module.
        

        quant_state_dict = {
            key[len("weight."):]: value
            for key, value in raw_state.items()
            if key.startswith("weight.")
        }
        quant_state = bnb_functional.QuantState.from_dict(
            quant_state_dict,
            device=raw_state["weight"].device,
        )

        canonical = {
            "weight": raw_state["weight"],
            "absmax": quant_state.absmax,
            "code": quant_state.code,
            "offset": quant_state.offset,
        }

        if quant_state.state2 is not None:
            canonical["state2_absmax"] = quant_state.state2.absmax
            canonical["state2_code"] = quant_state.state2.code
            canonical["state2_offset"] = quant_state.state2.offset

        return canonical

    @classmethod
    def _expert_staging_size(cls, expert) -> int:
        total = 0

        for name in cls._PROJECTIONS:
            state = cls._projection_state(expert, name)
            total += sum(cls._tensor_bytes(tensor) for tensor in state.values())

        return total

    @staticmethod
    def _stage_tensor(tensor: torch.Tensor) -> torch.Tensor:
        pinned = torch.empty_like(tensor, pin_memory=True)
        pinned.copy_(tensor)
        return pinned

    def stage_expert(
        self,
        slot: StagingSlot,
        expert,
        layer_id: int,
        expert_id: int,
    ) -> None:
        if not slot.in_use:
            raise RuntimeError("Slot must be acquired before staging")

        old_tensors = slot.tensors
        staged: dict[str, dict[str, torch.Tensor | None]] = {}

        for name in self._PROJECTIONS:
            source_state = self._projection_state(expert, name)
            old_state = old_tensors.get(name, {})

            state: dict[str, torch.Tensor | None] = {}
            for key, source in source_state.items():
                existing = old_state.get(key)
                state[key] = self._copy_into_pinned(source, existing)

            staged[name] = state

        slot.tensors = staged
        slot.expert_key = (layer_id, expert_id)

    @staticmethod
    def _copy_into_pinned(
        source: torch.Tensor | None,
        existing: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if source is None:
            return None

        if existing is None:
            existing = torch.empty_like(source, pin_memory=True)

        if existing.shape != source.shape or existing.dtype != source.dtype:
            raise ValueError(
                "Existing staging buffer does not match source tensor"
            )

        existing.copy_(source)
        return existing
