from calc.dtw_distance import get_dtw_distance
from pose.landmarks import get_landmarks_vectors


def get_pushup_pose_result(
    input_pose_landmarks, image_pose_landmarks, image_joint_angles, input_joint_angles
) -> dict:
    image_key_points_vector_x, image_key_points_vector_y = get_landmarks_vectors(
        image_pose_landmarks
    )
    input_key_points_vector_x, input_key_points_vector_y = get_landmarks_vectors(
        input_pose_landmarks
    )

    x_distance = get_dtw_distance(input_key_points_vector_x, image_key_points_vector_x)
    # y_distance = get_dtw_distance(input_key_points_vector_y, image_key_points_vector_y)
    return {
        "comparison_score": x_distance,
    }
