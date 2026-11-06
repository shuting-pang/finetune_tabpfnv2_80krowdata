"""Deterministic epoch-wise row resampling for TabPFN-2 fine-tuning."""

from __future__ import annotations

import numpy as np


def epoch_batches(n_rows: int, max_rows: int, seed: int, epoch: int) -> list[np.ndarray]:
    """Return a new partition of every training row, without replacement.

    Each epoch covers the full training set once. Shuffling changes which rows
    meet in a <= max_rows TabPFN context across epochs.
    """
    if n_rows < 1 or max_rows < 2:
        raise ValueError("n_rows must be positive and max_rows must be at least 2")
    if epoch < 0:
        raise ValueError("epoch must be nonnegative")
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    order = rng.permutation(n_rows)
    return [order[start : start + max_rows] for start in range(0, n_rows, max_rows)]
