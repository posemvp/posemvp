from exercises_detector.constants import ExerciseType
from exercises_detector.detector import Detector, YogaDetector, WorkoutDetector


class DetectorFactory:
    @staticmethod
    def get_type(_type: str) -> Detector:
        if _type == ExerciseType.YOGA.value:
            return YogaDetector()
        elif _type == ExerciseType.WORKOUT.value:
            return WorkoutDetector()
