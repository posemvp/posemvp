from exercises_detector.constants import ExerciseName, ExerciseType
from exercises_detector.detector_factory import DetectorFactory


def compare_pose(image_pose_landmarks, input_pose_landmarks):
    detector = DetectorFactory.get_type(ExerciseType.YOGA.value).get(
        ExerciseName.MOUNTAIN_POSE.value
    )
    detector.get_result(image_pose_landmarks, input_pose_landmarks)
