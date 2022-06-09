from calc.angle import find_angle
from exercises_detector.detector import WorkoutDetector


class PushUpDetector(WorkoutDetector):
    def _get_result(self, poses) -> str:
        feedbackList = []
        if (
            poses.pose_landmarks.landmark[11].visibility > 0.8
            and poses.pose_landmarks.landmark[13].visibility > 0.8
            and poses.pose_landmarks.landmark[15].visibility > 0.8
            and poses.pose_landmarks.landmark[23].visibility > 0.8
            and poses.pose_landmarks.landmark[25].visibility > 0.8
        ) or (
            poses.pose_landmarks.landmark[12].visibility > 0.8
            and poses.pose_landmarks.landmark[14].visibility > 0.8
            and poses.pose_landmarks.landmark[16].visibility > 0.8
            and poses.pose_landmarks.landmark[24].visibility > 0.8
            and poses.pose_landmarks.landmark[26].visibility > 0.8
        ):
            right_hand_shoulder_angle = find_angle(
                poses.pose_landmarks.landmark[11],
                poses.pose_landmarks.landmark[13],
                poses.pose_landmarks.landmark[15],
            )
            left_hand_shoulder_angle = find_angle(
                poses.pose_landmarks.landmark[12],
                poses.pose_landmarks.landmark[14],
                poses.pose_landmarks.landmark[16],
            )
            right_leg_hip_angle = find_angle(
                poses.pose_landmarks.landmark[11],
                poses.pose_landmarks.landmark[23],
                poses.pose_landmarks.landmark[25],
            )
            left_leg_hip_angle = find_angle(
                poses.pose_landmarks.landmark[12],
                poses.pose_landmarks.landmark[24],
                poses.pose_landmarks.landmark[26],
            )
            if right_hand_shoulder_angle < 45 or left_hand_shoulder_angle < 45:
                feedbackList.append("Dont go to lower")
            if left_leg_hip_angle < 150 or right_leg_hip_angle < 150:
                feedbackList.append("Straight your waist")
        return str(feedbackList)
