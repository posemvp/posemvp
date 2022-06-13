from calc.dtw_distance import get_dtw_distance


def get_pose_result(input_pose_landmarks, image_pose_landmarks):
    input_vector_x = []
    input_vector_y = []
    image_vector_x = []
    image_vector_y = []
    for key, key_point in input_pose_landmarks.items():
        input_vector_x.append(key_point.x)
        input_vector_y.append(key_point.y)

    for key, key_point in image_pose_landmarks.items():
        image_vector_x.append(key_point.x)
        image_vector_y.append(key_point.y)

    xDistance = get_dtw_distance(input_vector_x, image_vector_x)
    yDistance = get_dtw_distance(input_vector_y, image_vector_y)
    print(f"x: {xDistance}, y: {yDistance}")
    # return f"x: {xDistance}, y: {yDistance}"
