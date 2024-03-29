from abc import ABC, abstractmethod
from typing import List
from exercises_detector.constants import ExerciseName
from exercises_detector.workout.pushup import get_pushup_pose_result
from exercises_detector.yoga.mountain_pose import get_mountain_pose_result
from exercises_detector.yoga.warrior_II_pose import get_warrior_II_pose_result
from pose.joint_angles import get_joint_angles


class Detector(ABC):
    @abstractmethod
    def get_result(
        self, image_pose_landmarks, input_pose_landmarks, image_joint_angles: dict
    ) -> dict:
        pass

    @staticmethod
    def get_joint_angles(landmarks) -> dict:
        return get_joint_angles(landmarks)

    @abstractmethod
    def get(self, name):
        pass


class YogaDetector(Detector):
    def get(self, name: str):
        if name == ExerciseName.MOUNTAIN_POSE.value:
            return MountainPoseDetector()
        if name == ExerciseName.WARRIOR_II_POSE.value:
            return WarriorIIPoseDetector()

    def get_result(
        self, image_pose_landmarks, input_pose_landmarks, image_joint_angles: dict
    ) -> dict:
        return self._get_pose_result(
            input_pose_landmarks, image_pose_landmarks, image_joint_angles
        )

    def _get_pose_result(
        self, input_pose_landmarks, image_pose_landmarks, image_joint_angles
    ) -> dict:
        pass


class MountainPoseDetector(YogaDetector):
    def _get_pose_result(
        self, input_pose_landmarks, image_pose_landmarks, image_joint_angles
    ) -> dict:
        return get_mountain_pose_result(input_pose_landmarks, image_pose_landmarks)


class WarriorIIPoseDetector(YogaDetector):
    def _get_pose_result(
        self, input_pose_landmarks, image_pose_landmarks, image_joint_angles
    ) -> dict:
        return get_warrior_II_pose_result(
            input_pose_landmarks,
            image_pose_landmarks,
            image_joint_angles,
            super(WarriorIIPoseDetector, self).get_joint_angles(input_pose_landmarks),
        )


class WorkoutDetector(Detector):
    def get_result(
        self, image_pose_landmarks, input_pose_landmarks, image_joint_angles: dict
    ) -> List[float]:
        pass

    def get(self, name):
        if name == ExerciseName.PUSH_UP.value:
            return PushUpDetector()


class PushUpDetector(WorkoutDetector):
    def _get_pose_result(
        self, input_pose_landmarks, image_pose_landmarks, image_joint_angles
    ) -> dict:
        return get_pushup_pose_result(
            input_pose_landmarks,
            image_pose_landmarks,
            image_joint_angles,
            super(PushUpDetector, self).get_joint_angles(input_pose_landmarks),
        )
