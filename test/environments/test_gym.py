import unittest

from rlib.environments.gym import GymEnvironment


class GymEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.env = GymEnvironment("Pendulum-v0")
