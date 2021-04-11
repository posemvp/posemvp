from typing import List
from utils.angle import find_angle
from getImagePose import image_pose_x, image_pose_y
from dtwCalculator import calculateDistance

down = False

def getResult(results) -> List[str]:
    vector_x = []
    vector_y = []

    for i in range(33):
        vector_x.append(results.pose_landmarks.landmark[i].x)
        vector_y.append(results.pose_landmarks.landmark[i].y)

    xDistance = calculateDistance(vector_x, image_pose_x)
    yDistance = calculateDistance(vector_y, image_pose_y)

    # print("xDistance: ", xDistance, " yDistance: ", yDistance)
    return []
            


        
