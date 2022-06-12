from pose.constants import LANDMARK_INDEX_TYPE_MAP, USELESS_POSES


class Landmark:
    def __init__(self, landmark_index, x, y, z, visibility):
        self.type = LANDMARK_INDEX_TYPE_MAP.get(landmark_index)
        self.x = x
        self.y = y
        self.z = z
        self.point2d = (x, y)
        self.visibility = visibility


def _remove_face_hand_pose_estimation(landmarks):
    for pose in USELESS_POSES:
        landmarks.pop(pose)
    return landmarks


def get_landmark_key_points(image, raw_landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_key_points = {}
    for index, raw_landmark in enumerate(raw_landmarks.landmark):
        landmark = Landmark(
            landmark_index=index,
            x=min(int(raw_landmark.x * image_width), image_width - 1),
            y=min(int(raw_landmark.y * image_height), image_height - 1),
            z=raw_landmark.z,
            visibility=raw_landmark.visibility,
        )
        landmark_key_points[landmark.type] = landmark
    landmark_key_points = _remove_face_hand_pose_estimation(landmark_key_points)
    return landmark_key_points
