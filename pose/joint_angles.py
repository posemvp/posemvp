import math


def __get_angle(A, B, C):
    AB = math.sqrt(math.pow(B.x - A.x, 2) + math.pow(B.y - A.y, 2))
    BC = math.sqrt(math.pow(B.x - C.x, 2) + math.pow(B.y - C.y, 2))
    CA = math.sqrt(math.pow(C.x - A.x, 2) + math.pow(C.y - A.y, 2))
    return math.acos((BC * BC + AB * AB - CA * CA) / (2 * BC * AB)) * 180 / math.pi


def get_joint_angles(landmarks) -> dict:
    return {
        "RIGHT_ELBOW_ANGLE": __get_angle(
            landmarks["RIGHT_WRIST"].normalizedPoint3d,
            landmarks["RIGHT_ELBOW"].normalizedPoint3d,
            landmarks["RIGHT_SHOULDER"].normalizedPoint3d,
        ),
        "RIGHT_SHOULDER_ANGLE": __get_angle(
            landmarks["RIGHT_ELBOW"].normalizedPoint3d,
            landmarks["RIGHT_SHOULDER"].normalizedPoint3d,
            landmarks["RIGHT_HIP"].normalizedPoint3d,
        ),
        "RIGHT_HIP_ANGLE": __get_angle(
            landmarks["RIGHT_SHOULDER"].normalizedPoint3d,
            landmarks["RIGHT_HIP"].normalizedPoint3d,
            landmarks["RIGHT_KNEE"].normalizedPoint3d,
        ),
        "RIGHT_KNEE_ANGLE": __get_angle(
            landmarks["RIGHT_HIP"].normalizedPoint3d,
            landmarks["RIGHT_KNEE"].normalizedPoint3d,
            landmarks["RIGHT_ANKLE"].normalizedPoint3d,
        ),
        "LEFT_ELBOW_ANGLE": __get_angle(
            landmarks["RIGHT_WRIST"].normalizedPoint3d,
            landmarks["RIGHT_ELBOW"].normalizedPoint3d,
            landmarks["RIGHT_SHOULDER"].normalizedPoint3d,
        ),
        "LEFT_SHOULDER_ANGLE": __get_angle(
            landmarks["LEFT_ELBOW"].normalizedPoint3d,
            landmarks["LEFT_SHOULDER"].normalizedPoint3d,
            landmarks["LEFT_HIP"].normalizedPoint3d,
        ),
        "LEFT_HIP_ANGLE": __get_angle(
            landmarks["LEFT_SHOULDER"].normalizedPoint3d,
            landmarks["LEFT_HIP"].normalizedPoint3d,
            landmarks["LEFT_KNEE"].normalizedPoint3d,
        ),
        "LEFT_KNEE_ANGLE": __get_angle(
            landmarks["LEFT_HIP"].normalizedPoint3d,
            landmarks["LEFT_KNEE"].normalizedPoint3d,
            landmarks["LEFT_ANKLE"].normalizedPoint3d,
        ),
    }
