import unittest

import numpy as np

from resampling import epoch_batches


class EpochBatchesTest(unittest.TestCase):
    def test_full_coverage_and_limit(self):
        batches = epoch_batches(80_000, 10_000, seed=42, epoch=1)
        self.assertEqual(len(batches), 8)
        self.assertTrue(all(len(batch) == 10_000 for batch in batches))
        self.assertTrue(np.array_equal(np.sort(np.concatenate(batches)), np.arange(80_000)))

    def test_epochs_change_companions_deterministically(self):
        first = epoch_batches(200, 50, seed=42, epoch=1)
        repeated = epoch_batches(200, 50, seed=42, epoch=1)
        second = epoch_batches(200, 50, seed=42, epoch=2)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(first, repeated)))
        self.assertFalse(np.array_equal(first[0], second[0]))


if __name__ == "__main__":
    unittest.main()
