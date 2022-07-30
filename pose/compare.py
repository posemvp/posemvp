from exercises_detector.constants import ExerciseName, ExerciseType
from exercises_detector.detector_factory import DetectorFactory
from pose.joint_angles import get_joint_angles


def compare_pose(image_pose_landmarks, input_pose_landmarks) -> dict:
    detector = DetectorFactory.get_type(ExerciseType.YOGA.value).get(
        ExerciseName.WARRIOR_II_POSE.value
    )
    return detector.get_result(
        image_pose_landmarks,
        input_pose_landmarks,
        get_joint_angles(image_pose_landmarks),
    )
