from typing import List
from exercises_detector.constants import ExerciseName, ExerciseType
from exercises_detector.detector_factory import DetectorFactory


def compare_pose(image_pose_landmarks, input_pose_landmarks) -> List[float]:
    detector = DetectorFactory.get_type(ExerciseType.YOGA.value).get(
        ExerciseName.WARRIOR_II_POSE.value
    )
    return detector.get_result(image_pose_landmarks, input_pose_landmarks)
