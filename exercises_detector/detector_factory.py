from enum import Enum

from exercises_detector.detector import Detector
from exercises_detector.workout.pushup import PushUpDetector
from exercises_detector.yoga.mountain_pose import MountainPoseDetector


class ExerciseType(Enum):
    YOGA = "YOGA"
    WORKOUT = "WORKOUT"
    PREGNANCY = "PREGNANCY"
    REHABILITATION = "REHABILITATION"


class ExerciseName(Enum):
    PUSH_UP = "PUSH_UP"
    MOUNTAIN_POSE = "MOUNTAIN_POSE"


class DetectorFactory:
    @staticmethod
    def get(name: str) -> Detector:
        if name == ExerciseName.PUSH_UP.value:
            return PushUpDetector()
        elif name == ExerciseName.MOUNTAIN_POSE.value:
            return MountainPoseDetector()
