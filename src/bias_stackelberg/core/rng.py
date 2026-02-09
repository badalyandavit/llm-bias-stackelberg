from __future__ import annotations

import hashlib


def stable_int_seed(seed: int, prompt: str) -> int:
    """
    Deterministic mixing of (seed, prompt) -> 32-bit int.
    Used to make mock generations reproducible across processes.
    """
    s = f"{seed}|{prompt}".encode()
    h = hashlib.sha256(s).digest()
    return int.from_bytes(h[:4], byteorder="big", signed=False)
