"""Geometry columns 5-7 of the unified feature specification, on a polygon whose values are known in closed form."""
import math
import unittest

import numpy as np

from secondlook.features.unified_features import UNIFIED_COLUMNS, shoelace, unified_geometry


class Geometry(unittest.TestCase):
    def test_columns(self):
        self.assertEqual(len(UNIFIED_COLUMNS), 11)
        self.assertEqual(UNIFIED_COLUMNS[5:8], ['log_rel_area', 'log_height', 'log_aspect'])

    def test_rectangle(self):
        P = np.array([[10, 10], [30, 10], [30, 20], [10, 20]])     # 20 x 10 pixels
        self.assertAlmostEqual(shoelace(P), 200.0)
        a, h, r = unified_geometry(P, (200, 100))                  # image 200 x 100
        self.assertAlmostEqual(a, math.log(200.0 / (200 * 100)))
        self.assertAlmostEqual(h, math.log(10.0))
        self.assertAlmostEqual(r, math.log(2.0))

    def test_degenerate_polygon_is_clamped(self):
        P = np.array([[5, 5], [5, 5], [5, 5], [5, 5]])
        a, h, r = unified_geometry(P, (100, 100))
        self.assertAlmostEqual(h, 0.0)                              # extent clamped to 1 pixel
        self.assertAlmostEqual(r, 0.0)
        self.assertAlmostEqual(a, math.log(1.0 / 10000))            # area clamped to 1


if __name__ == '__main__':
    unittest.main()
