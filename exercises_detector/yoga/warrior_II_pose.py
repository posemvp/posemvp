from typing import List
import numpy as np
from calc.dtw_distance import get_dtw_distance


def get_warrior_II_pose_result(
    input_pose_landmarks, image_pose_landmarks
) -> List[float]:
    input_vector_x = []
    input_vector_y = []
    image_vector_x = []
    image_vector_y = []

    for key, key_point in input_pose_landmarks.items():
        input_vector_x.append(key_point.to_dict()["normalized_point3d"]["x"])
        input_vector_y.append(key_point.to_dict()["normalized_point3d"]["y"])

    for key, key_point in image_pose_landmarks.items():
        image_vector_x.append(key_point.to_dict()["normalized_point3d"]["x"])
        image_vector_y.append(key_point.to_dict()["normalized_point3d"]["y"])

    input_x_points = np.array(input_vector_x)
    input_y_points = np.array(input_vector_y)
    image_x_points = np.array(image_vector_x)
    image_y_points = np.array(image_vector_y)

    x_distance = get_dtw_distance(input_x_points, image_x_points)
    # y_distance = get_dtw_distance(input_y_points, image_y_points)
    return [x_distance]
