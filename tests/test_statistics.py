"""Bootstrap and F1 helpers: determinism and closed-form values."""
import unittest

import numpy as np

from secondlook.evaluation.common import BOOT, boot_rates, ci, det_rates, index_matrix
from secondlook.studies.learning_curve import f1c


class Statistics(unittest.TestCase):
    def test_index_matrix_is_deterministic(self):
        a, b = index_matrix(37), index_matrix(37)
        self.assertEqual(a.shape, (BOOT, 37))
        self.assertTrue((a == b).all())
        self.assertTrue(a.min() >= 0 and a.max() < 37)

    def test_f1_from_counts(self):
        c = np.array([[3, 4, 5], [2, 2, 1]])        # per image: correct, predicted, ground truth
        self.assertAlmostEqual(f1c(c), 200.0 * 5 / (6 + 6))

    def test_detection_rates(self):
        c = np.zeros((2, 7), np.int64)
        c[:, 4], c[:, 5], c[:, 6] = [3, 1], [4, 2], [5, 3]
        r = det_rates(c)
        self.assertAlmostEqual(r['precision'], 100 * 4 / 6)
        self.assertAlmostEqual(r['recall'], 100 * 4 / 8)
        self.assertAlmostEqual(r['f'], 200 * 4 / 14)

    def test_bootstrap_of_identity_resample(self):
        c = np.zeros((3, 7), np.int64)
        c[:, 4], c[:, 5], c[:, 6] = [1, 2, 3], [2, 2, 4], [2, 3, 3]
        M = np.tile(np.arange(3), (5, 1))
        r = boot_rates(c, M)
        self.assertTrue(np.allclose(r['f'], det_rates(c)['f']))

    def test_percentile_interval(self):
        lo, hi = ci(np.arange(101.0))
        self.assertAlmostEqual(lo, 2.5)
        self.assertAlmostEqual(hi, 97.5)


if __name__ == '__main__':
    unittest.main()
