from abc import ABC, abstractmethod
from exercises_detector.constants import ExerciseName
from exercises_detector.yoga.mountain_pose import get_pose_result


class Detector(ABC):
    @abstractmethod
    def get_result(self, image_pose_landmarks, input_pose_landmarks) -> str:
        pass

    @abstractmethod
    def get(self, name):
        pass


class YogaDetector(Detector):
    def get(self, name: str):
        if name == ExerciseName.MOUNTAIN_POSE.value:
            return MountainPoseDetector()
        if name == ExerciseName.WARRIOR_II_POSE.value:
            return WarriorIIPoseDetector()

    def get_result(self, image_pose_landmarks, input_pose_landmarks) -> str:
        return self._get_pose_result(input_pose_landmarks, image_pose_landmarks)

    def _get_pose_result(self, input_pose_landmarks, image_pose_landmarks) -> str:
        pass


class MountainPoseDetector(YogaDetector):
    def _get_pose_result(self, input_pose_landmarks, image_pose_landmarks) -> str:
        return get_pose_result(input_pose_landmarks, image_pose_landmarks)


class WarriorIIPoseDetector(YogaDetector):
    def _get_pose_result(self, input_pose_landmarks, image_pose_landmarks) -> str:
        return get_pose_result(input_pose_landmarks, image_pose_landmarks)


class WorkoutDetector(Detector):
    def get_result(self, image_pose_landmarks, input_pose_landmarks) -> str:
        pass

    def get(self, name):
        if name == ExerciseName.PUSH_UP.value:
            return PushUpDetector()


class PushUpDetector(WorkoutDetector):
    def get_result(self, pose_name, input_pose_landmarks) -> str:
        return ""
