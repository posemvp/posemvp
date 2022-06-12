from enum import Enum
import cv2 as cv

_GREEN_COLOR_RGB = (0, 255, 0)
_RED_COLOR_RGB = (0, 255, 0)
_KEYPOINT_RADIUS = 5


class EstimationResultType(Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0), 2)
    return image


def draw_text(image, landmark_x, landmark_y, landmark_z):
    cv.putText(
        image,
        "z:" + str(round(landmark_z, 3)),
        (landmark_x - 10, landmark_y - 10),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        _GREEN_COLOR_RGB,
        1,
        cv.LINE_AA,
    )


def draw_circle(image, landmark_x, landmark_y, _type=EstimationResultType.SUCCESS.value):
    if _type == EstimationResultType.SUCCESS.value:
        cv.circle(image, (landmark_x, landmark_y), _KEYPOINT_RADIUS, _GREEN_COLOR_RGB, 2)
    elif _type == EstimationResultType.FAILED.value:
        cv.circle(image, (landmark_x, landmark_y), _KEYPOINT_RADIUS, _RED_COLOR_RGB, 2)


def draw_line(image, landmark_x, landmark_y, _type=EstimationResultType.SUCCESS.value):
    if _type == EstimationResultType.SUCCESS.value:
        cv.line(image, landmark_x, landmark_y, _GREEN_COLOR_RGB, 2)
    elif _type == EstimationResultType.FAILED.value:
        cv.line(image, landmark_x, landmark_y, _RED_COLOR_RGB, 2)
