from typing import List
from calc.angle import find_angle

down = False


def getResult(results) -> List[str]:
    feedbackList = []
    if (
        results.pose_landmarks.landmark[11].visibility > 0.8
        and results.pose_landmarks.landmark[13].visibility > 0.8
        and results.pose_landmarks.landmark[15].visibility > 0.8
        and results.pose_landmarks.landmark[23].visibility > 0.8
        and results.pose_landmarks.landmark[25].visibility > 0.8
    ) or (
        results.pose_landmarks.landmark[12].visibility > 0.8
        and results.pose_landmarks.landmark[14].visibility > 0.8
        and results.pose_landmarks.landmark[16].visibility > 0.8
        and results.pose_landmarks.landmark[24].visibility > 0.8
        and results.pose_landmarks.landmark[26].visibility > 0.8
    ):
        right_hand_shoulder_angle = find_angle(
            results.pose_landmarks.landmark[11],
            results.pose_landmarks.landmark[13],
            results.pose_landmarks.landmark[15],
        )
        left_hand_shoulder_angle = find_angle(
            results.pose_landmarks.landmark[12],
            results.pose_landmarks.landmark[14],
            results.pose_landmarks.landmark[16],
        )
        right_leg_hip_angle = find_angle(
            results.pose_landmarks.landmark[11],
            results.pose_landmarks.landmark[23],
            results.pose_landmarks.landmark[25],
        )
        left_leg_hip_angle = find_angle(
            results.pose_landmarks.landmark[12],
            results.pose_landmarks.landmark[24],
            results.pose_landmarks.landmark[26],
        )
        if right_hand_shoulder_angle < 45 or left_hand_shoulder_angle < 45:
            feedbackList.append("Dont go to lower")
        if left_leg_hip_angle < 150 or right_leg_hip_angle < 150:
            feedbackList.append("Straight your waist")
    return feedbackList
