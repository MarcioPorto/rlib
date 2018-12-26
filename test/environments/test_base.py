import unittest

from rlib.environments.base import BaseEnvironment


class BaseEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.env = BaseEnvironment()
