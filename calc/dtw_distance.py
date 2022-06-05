from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


def get_dtw_distance(vec1, vec2):
    distance, path = fastdtw(vec1, vec2, dist=euclidean)
    return distance
