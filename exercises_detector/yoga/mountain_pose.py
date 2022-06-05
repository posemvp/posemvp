from typing import List
from pose.get_image_pose import image_pose_x, image_pose_y
from calc.dtw_calculator import calculateDistance

down = False


def get_result(results) -> List[str]:
    vector_x = []
    vector_y = []
    for i in range(33):
        vector_x.append(results.pose_landmarks.landmark[i].x)
        vector_y.append(results.pose_landmarks.landmark[i].y)

    xDistance = calculateDistance(vector_x, image_pose_x)
    yDistance = calculateDistance(vector_y, image_pose_y)
    return []
