"""Deterministic random-stream construction for simulations."""

from typing import Optional, Union
import numpy as np


Seed = Optional[Union[int, np.random.Generator]]

_STREAMS = {
    "traffic": 0,
    "Pure ALOHA": 1,
    "Slotted ALOHA": 2,
    "1-Persistent CSMA": 3,
}


def stream_rng(
    seed: Seed,
    stream: str,
    supplied_rng: Optional[np.random.Generator] = None,
) -> np.random.Generator:
    """Return a reproducible RNG for a named traffic or protocol stream."""
    if supplied_rng is not None:
        return supplied_rng
    if isinstance(seed, np.random.Generator):
        return seed
    if stream not in _STREAMS:
        raise ValueError(f"Unknown random stream: {stream}")
    return np.random.default_rng(
        np.random.SeedSequence(seed, spawn_key=(_STREAMS[stream],))
    )
