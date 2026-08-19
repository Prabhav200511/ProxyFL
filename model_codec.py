"""Opaque model-state serialization used inside authenticated network payloads."""

from __future__ import annotations

from io import BytesIO
from typing import Mapping

import torch


def serialize_weights(weights: Mapping[str, torch.Tensor]) -> bytes:
    """Serialize a tensor-only state dictionary after it has been locally produced."""
    stream = BytesIO()
    # Copy tensors to CPU so a receiving process/node is not coupled to a GPU.
    safe_state = {name: tensor.detach().cpu() for name, tensor in weights.items()}
    torch.save(safe_state, stream)
    return stream.getvalue()


def deserialize_weights(payload: bytes) -> dict[str, torch.Tensor]:
    """Load only tensors, never arbitrary pickle globals from network content."""
    if not isinstance(payload, bytes):
        raise ValueError("model payload must be bytes")
    stream = BytesIO(payload)
    try:
        state = torch.load(stream, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("ProxyFL requires a PyTorch version with weights_only loading") from exc
    if not isinstance(state, dict) or not all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor)
        for name, tensor in state.items()
    ):
        raise ValueError("model payload is not a tensor-only state dictionary")
    return state
