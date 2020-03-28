import numpy


def range_parameter(three_tuple):
    yield from three_tuple


import unittest

class ListToolsTest(unittest.TestCase):
    def test_third_fractal(self):
        s_tuple = (10,40,70)
        print (list(range_parameter(s_tuple)))


if __name__ == '__main__':
    unittest.main()


