import numpy as np
from numpy import ndarray
from typing import Tuple
from pose.constants import LANDMARK_INDEX_TYPE_MAP, USELESS_POSES


class Point2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class NormalizedPoint3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def to_dict(self):
        return {"x": self.x, "y": self.y, "z": self.z}


class Landmark:
    def __init__(self, landmark_index, point2d, normalizedPoint3d, visibility):
        self.type = LANDMARK_INDEX_TYPE_MAP.get(landmark_index)
        self.point2d = (point2d.x, point2d.y)
        self.normalizedPoint3d = normalizedPoint3d
        self.visibility = visibility

    def to_dict(self):
        return {
            "type": self.type,
            "normalized_point3d": self.normalizedPoint3d.to_dict(),
            "visibility": self.visibility,
        }


def _remove_useless_pose_estimation(landmarks):
    for pose in USELESS_POSES:
        landmarks.pop(pose)
    return landmarks


def _get_normalized_landmark_point(landmark) -> NormalizedPoint3D:
    normalize_coordinate = (landmark.x, landmark.y)
    normalize_coordinate = normalize_coordinate / np.linalg.norm(normalize_coordinate)
    return NormalizedPoint3D(
        normalize_coordinate[0], normalize_coordinate[1], landmark.z
    )


def get_landmark_key_points(image, raw_landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_key_points = {}
    for index, raw_landmark in enumerate(raw_landmarks.landmark):
        landmark = Landmark(
            landmark_index=index,
            point2d=Point2D(
                x=min(int(raw_landmark.x * image_width), image_width - 1),
                y=min(int(raw_landmark.y * image_height), image_height - 1),
            ),
            normalizedPoint3d=_get_normalized_landmark_point(raw_landmark),
            visibility=raw_landmark.visibility,
        )
        landmark_key_points[landmark.type] = landmark
    landmark_key_points = _remove_useless_pose_estimation(landmark_key_points)
    return landmark_key_points


def get_landmarks_vectors(landmarks) -> Tuple[ndarray, ndarray]:
    vector_x = []
    vector_y = []
    for key, key_point in landmarks.items():
        vector_x.append(key_point.normalizedPoint3d.x)
        vector_y.append(key_point.normalizedPoint3d.y)
    return np.array(vector_x), np.array(vector_y)
