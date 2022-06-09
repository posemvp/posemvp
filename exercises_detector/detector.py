from abc import ABC, abstractmethod
from typing import Tuple
import mediapipe as mp


class Detector(ABC):
    @abstractmethod
    def _get_result(self, poses) -> str:
        pass


class YogaDetector(Detector):
    @abstractmethod
    def _get_image(self):
        pass

    def _get_result(self, poses) -> str:
        image_x, image_y = self._get_image_pose_score
        return self._get_pose_result(poses, image_x, image_y)

    @abstractmethod
    def _get_pose_result(self, poses, image_x, image_y) -> str:
        pass

    @property
    def _get_image_pose_score(self) -> Tuple[list, list]:
        poses = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ).process(self._get_image())
        image_pose_x = []
        image_pose_y = []
        for i in range(33):
            image_pose_x.append(poses.pose_landmarks.landmark[i].x)
            image_pose_y.append(poses.pose_landmarks.landmark[i].y)
        return image_pose_x, image_pose_y


class WorkoutDetector(Detector):
    def _get_result(self, poses) -> str:
        pass


class PregnancyDetector(Detector):
    def _get_result(self, poses) -> str:
        pass


class RehabilitationDetector(Detector):
    def _get_result(self, poses) -> str:
        pass
