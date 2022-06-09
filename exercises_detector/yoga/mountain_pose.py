from exercises_detector.detector import YogaDetector
from calc.dtw_distance import get_dtw_distance
import cv2 as cv


class MountainPoseDetector(YogaDetector):
    def _get_image(self):
        image = cv.imread(f"pose_images/{self.__class__.__name__}.jpg")
        return image

    def _get_pose_result(self, poses, image_x, image_y) -> str:
        vector_x = []
        vector_y = []
        for i in range(33):
            vector_x.append(poses.pose_landmarks.landmark[i].x)
            vector_y.append(poses.pose_landmarks.landmark[i].y)
        xDistance = get_dtw_distance(vector_x, image_x)
        yDistance = get_dtw_distance(vector_y, image_y)
        return f"x: {xDistance}, y: {yDistance}"
