def get_pose_result(input_pose_landmarks, image_pose_landmarks):
    input_vector_x = []
    input_vector_y = []
    image_vector_x = []
    image_vector_y = []
    for key, key_point in input_pose_landmarks.items():
        input_vector_x.append(key_point.to_dict())
        input_vector_y.append(key_point.to_dict())

    for key, key_point in image_pose_landmarks.items():
        image_vector_x.append(key_point.to_dict())
        image_vector_y.append(key_point.to_dict())

    print("====================")
    print(image_vector_x)
    print(input_vector_x)
    print("--------------------")
    print(image_vector_y)
    print(input_vector_y)
    print("====================")
