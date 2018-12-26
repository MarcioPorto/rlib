import unittest

from rlib.shared.replay_buffer import ReplayBuffer


class ReplayBufferTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.replay_buffer = ReplayBuffer(10, self.batch_size, "cpu")
        self.populate_replay_buffer()

    def populate_replay_buffer(self, n=5):
        for _ in range(n):
            self.replay_buffer.add(
                0.0, 0.0, 0.0, 0.0, 0.0
            )

    def test_add(self):
        l1 = len(self.replay_buffer)
        self.replay_buffer.add(
            0.0, 0.0, 0.0, 0.0, 0.0
        )
        l2 = len(self.replay_buffer)
        self.assertNotEqual(l1, l2)

    def test_sample(self):
        s, a, r, ns, d = self.replay_buffer.sample()
        self.assertEqual(s.shape[0], self.batch_size)
        self.assertEqual(a.shape[0], self.batch_size)
        self.assertEqual(r.shape[0], self.batch_size)
        self.assertEqual(ns.shape[0], self.batch_size)
        self.assertEqual(d.shape[0], self.batch_size)
