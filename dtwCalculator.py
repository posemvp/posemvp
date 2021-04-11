import string
import random
import numpy as np

from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

def calculateDistance(vec1, vec2):
    image_name = ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k = 10))
    image_file_name = str(image_name)
    distance = dtw.distance(vec1, vec2)
    # path = dtw.warping_path(vec1, vec2)
    # dtwvis.plot_warping(vec1, vec2, path, filename=f"dtwResultGraphs/mountainPose/{image_file_name}.png")
    return distance