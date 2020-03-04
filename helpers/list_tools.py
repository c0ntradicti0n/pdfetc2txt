from collections import defaultdict


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
