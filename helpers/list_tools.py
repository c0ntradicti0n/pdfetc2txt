import logging
from collections import defaultdict

import numpy


def find_repeating(lst, count=2):
    ret = []
    counts = [None] * len(lst)
    for i in lst:
        if counts[i] is None:
            counts[i] = i
        elif i == counts[i]:
            ret += [i]
            if len(ret) == count:
                return ret


def reverse_dict_of_lists(d):
    reversed_dict = defaultdict(list)
    for key, values in d.items():
        for value in values:
            reversed_dict[value].append(key)
    return reversed_dict

def threewise(iterable):
    "s -> (s0,s1,s2), (s1,s2, s3), (s2, s3, s4), ..."
    iterable = list(iterable)
    l = len(iterable)
    for i in range(1,l-1):
        yield iterable[i-1], iterable[i], iterable[i+1]


def third_fractal(s_value, s_tuple):
    """   [10, 40, 70] ->

          10- >  [-10.  10.  30.]
          40 ->  [ 20. 40. 60.]
          70 ->  [ 50. 70. 90.]

    """
    s1, s2, s3 = s_tuple
    new_range = s_value + 2/3 * (numpy.array(s_tuple) - s2 )
    return new_range

import unittest

class ListToolsTest(unittest.TestCase):
    def test_third_fractal(self):
        s_tuple = (10,40,70)
        for s in s_tuple:
            new_range = third_fractal(s, s_tuple)
            print (new_range)
            s1, s2, s3 = s_tuple
            ds = s3 - s1
            new_len = ds * 2 / 3
            logging.error(f" {s_tuple} is now {new_range}")
            #assert new_range.max() - new_range.min() == new_len
            #assert new_range[0] >= s1 and new_range[2] <= s3



if __name__ == '__main__':
    unittest.main()


