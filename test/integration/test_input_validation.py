from unittest import TestCase
from clustintime.clustintime import clustintime


class TestInputValidation(TestCase):

    def test_not_a_str(self):
        with self.assertRaises(ValueError):
            clustintime(data_file=1, mask_file='')
