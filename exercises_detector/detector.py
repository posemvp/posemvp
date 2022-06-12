import mediapipe as mp
import cv2 as cv
from abc import ABC, abstractmethod
from exercises_detector.constants import ExerciseName
from pose.landmarks import get_landmark_key_points


class Detector(ABC):
    @abstractmethod
    def get_result(self, pose_name, input_pose_landmarks) -> str:
        pass

    @abstractmethod
    def get(self, name):
        pass


class YogaDetector(Detector):
    def get(self, name: str):
        if name == ExerciseName.MOUNTAIN_POSE.value:
            return MountainPoseDetector()

    @staticmethod
    def _get_image(pose_name):
        image = cv.imread(f"exercises_detector/yoga/pose_images/{pose_name}.jpg")
        return image

    def _get_image_pose_score(self, pose_name):
        image = self._get_image(pose_name)
        pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ).process(image)
        pose_landmarks = pose.pose_landmarks
        if pose_landmarks is not None:
            return get_landmark_key_points(image, pose_landmarks)

    def get_result(self, pose_name, input_pose_landmarks) -> str:
        image_pose_landmarks = self._get_image_pose_score(pose_name)
        return self._get_pose_result(input_pose_landmarks, image_pose_landmarks)

    def _get_pose_result(self, input_pose_landmarks, image_pose_landmarks) -> str:
        pass


class MountainPoseDetector(YogaDetector):
    def _get_pose_result(self, input_pose_landmarks, image_pose_landmarks) -> str:
        return f"x:"


class WorkoutDetector(Detector):
    def get_result(self, pose_name, input_pose_landmarks) -> str:
        pass

    def get(self, name):
        if name == ExerciseName.PUSH_UP.value:
            return PushUpDetector()


class PushUpDetector(WorkoutDetector):
    def get_result(self, pose_name, input_pose_landmarks) -> str:
        return ""
