import logging
import math
import array
import numpy as np

from ed import distance

DTYPE = np.double
argmin = np.argmin
array_min = np.min
array_max = np.max

logger = logging.getLogger("dtw.distance")

inf = float("inf")


def ub_euclidean(s1, s2):
    """See ed.euclidean_distance"""
    return distance(s1, s2)


def dtwdistance(
    s1,
    s2,
    window=None,
    max_dist=None,
    max_step=None,
    max_length_diff=None,
    penalty=None,
    psi=None,
    use_c=False,
    use_pruning=False,
    only_ub=False,
):
    """
    Dynamic Time Warping.
    This function keeps a compact matrix, not the full warping paths matrix.
    :param s1: Firstx sequence
    :param s2: Second sequence
    :param window: Only allow for maximal shifts from the two diagonals smaller than this number.
        It includes the diagonal, meaning that an Euclidean distance is obtained by setting window=1.
    :param max_dist: Stop if the returned values will be larger than this value
    :param max_step: Do not allow steps larger than this value
    :param max_length_diff: Return infinity if length of two series is larger
    :param penalty: Penalty to add if compression or expansion is applied
    :param psi: Psi relaxation parameter (ignore start and end of matching).
        Useful for cyclical series.
    :param use_c: Use fast pure c compiled functions
    :param use_pruning: Prune values based on Euclidean distance.
        This is the same as passing ub_euclidean() to max_dist
    :param only_ub: Only compute the upper bound (Euclidean).
    Returns: DTW distance
    """
    # if use_c:
    #     if dtw_cc is None:
    #         logger.warning("C-library not available, using the Python version")
    #     else:
    #         return distance_fast(s1, s2, window,
    #                              max_dist=max_dist,
    #                              max_step=max_step,
    #                              max_length_diff=max_length_diff,
    #                              penalty=penalty,
    #                              psi=psi,
    #                              use_pruning=use_pruning,
    #                              only_ub=only_ub)
    r, c = len(s1), len(s2)
    if max_length_diff is not None and abs(r - c) > max_length_diff:
        return inf
    if window is None:
        window = max(r, c)
    if not max_step:
        max_step = inf
    else:
        max_step *= max_step
    if use_pruning or only_ub:
        max_dist = ub_euclidean(s1, s2) ** 2
        if only_ub:
            return max_dist
    elif not max_dist:
        max_dist = inf
    else:
        max_dist *= max_dist

    if not penalty:
        penalty = 0
    else:
        penalty *= penalty
    if psi is None:
        psi = 0
    length = min(c + 1, abs(r - c) + 2 * (window - 1) + 1 + 1 + 1)
    dtw = array.array("d", [inf] * (2 * length))
    sc = 0
    ec = 0
    ec_next = 0
    smaller_found = False
    for i in range(psi + 1):
        dtw[i] = 0
    skip = 0
    i0 = 1
    i1 = 0
    psi_shortest = inf
    for i in range(r):
        skipp = skip
        skip = max(0, i - max(0, r - c) - window + 1)
        i0 = 1 - i0
        i1 = 1 - i1
        for ii in range(i1 * length, i1 * length + length):
            dtw[ii] = inf
        j_start = max(0, i - max(0, r - c) - window + 1)
        j_end = min(c, i + max(0, c - r) + window)
        if sc > j_start:
            j_start = sc
        smaller_found = False
        ec_next = i
        if length == c + 1:
            skip = 0
        if psi != 0 and j_start == 0 and i < psi:
            dtw[i1 * length] = 0
        for j in range(j_start, j_end):
            d = (s1[i] - s2[j]) ** 2
            if d > max_step:
                continue
            assert j + 1 - skip >= 0
            assert j - skipp >= 0
            assert j + 1 - skipp >= 0
            assert j - skip >= 0
            dtw[i1 * length + j + 1 - skip] = d + min(
                dtw[i0 * length + j - skipp],
                dtw[i0 * length + j + 1 - skipp] + penalty,
                dtw[i1 * length + j - skip] + penalty,
            )
            if dtw[i1 * length + j + 1 - skip] > max_dist:
                if not smaller_found:
                    sc = j + 1
                if j >= ec:
                    break
            else:
                smaller_found = True
                ec_next = j + 1
        ec = ec_next
        if psi != 0 and j_end == len(s2) and len(s1) - 1 - i <= psi:
            psi_shortest = min(psi_shortest, dtw[i1 * length + length - 1])
    if psi == 0:
        d = dtw[i1 * length + min(c, c + window - 1) - skip]
    else:
        ic = min(c, c + window - 1) - skip
        vc = dtw[i1 * length + ic - psi : i1 * length + ic + 1]
        d = min(array_min(vc), psi_shortest)
    if max_dist and d > max_dist:
        d = inf
    d = math.sqrt(d)
    return d
