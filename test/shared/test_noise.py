import unittest

import numpy as np
from numpy import testing as nptesting

from rlib.shared.noise import OUNoise


class OUNoiseTest(unittest.TestCase):
    def setUp(self):
        self.noise = OUNoise(size=4, seed=0)

    def test_reset(self):
        self.noise.sample()
        s1 = self.noise.state
        self.noise.reset()
        s2 = self.noise.state
        nptesting.assert_equal(np.any(np.not_equal(s1, s2)), True)

    def test_sample(self):
        s1 = self.noise.state
        self.noise.sample()
        s2 = self.noise.state
        nptesting.assert_equal(np.any(np.not_equal(s1, s2)), True)
