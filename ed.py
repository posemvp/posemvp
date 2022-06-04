import math


def distance(s1, s2):
    """Euclidean distance between two sequences. Supports different lengths.
    If the two series differ in length, compare the last element of the shortest series
    to the remaining elements in the longer series. This is compatible with Euclidean
    distance being used as an upper bound for DTW.
    :param s1: Sequence of numbers
    :param s2: Sequence of numbers
    :return: Euclidean distance
    """
    n = min(len(s1), len(s2))
    ub = 0
    for v1, v2 in zip(s1, s2):
        ub += (v1 - v2) ** 2
    if len(s1) > len(s2):
        v2 = s2[n - 1]
        for v1 in s1[n:]:
            ub += (v1 - v2) ** 2
    elif len(s1) < len(s2):
        v1 = s1[n - 1]
        for v2 in s2[n:]:
            ub += (v1 - v2) ** 2
    return math.sqrt(ub)
