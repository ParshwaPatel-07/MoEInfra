import torch

from model.expert import QuantizedMixtralExpert


def test_prequantized_roundtrip():
    torch.manual_seed(0)

    hidden_size = 16
    intermediate_size = 32

    w1 = torch.randn(
        intermediate_size,
        hidden_size,
        dtype=torch.bfloat16,
    )
    w2 = torch.randn(
        hidden_size,
        intermediate_size,
        dtype=torch.bfloat16,
    )
    w3 = torch.randn(
        intermediate_size,
        hidden_size,
        dtype=torch.bfloat16,
    )

    # Build the normal expert and quantize it.
    reference_expert = QuantizedMixtralExpert(
        w1=w1,
        w2=w2,
        w3=w3,
    ).cuda()

    torch.cuda.synchronize()

    x = torch.randn(
        2,
        hidden_size,
        dtype=torch.float16,
        device="cuda",
    )

    with torch.no_grad():
        reference_output = reference_expert(x)

    # Serialize the already-quantized Linear4bit state.
    states = {}

    for name in ("w1", "w2", "w3"):
        states[name] = {
            key: value.detach().cpu().contiguous()
            for key, value in getattr(
                reference_expert,
                name,
            ).state_dict().items()
        }

    # Reconstruct directly from the pre-quantized state.
    reloaded_expert = QuantizedMixtralExpert.from_prequantized(
        w1_state=states["w1"],
        w2_state=states["w2"],
        w3_state=states["w3"],
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    ).cuda()

    torch.cuda.synchronize()

    with torch.no_grad():
        reloaded_output = reloaded_expert(x)

    assert torch.equal(
        reference_output,
        reloaded_output,
    )