from pose.get_image_pose import image_pose_x, image_pose_y
from calc.dtw_distance import get_dtw_distance

down = False


def get_result(results) -> str:
    vector_x = []
    vector_y = []
    for i in range(33):
        vector_x.append(results.pose_landmarks.landmark[i].x)
        vector_y.append(results.pose_landmarks.landmark[i].y)

    xDistance = get_dtw_distance(vector_x, image_pose_x)
    yDistance = get_dtw_distance(vector_y, image_pose_y)
    return f'x: {xDistance}, y: {yDistance}'
