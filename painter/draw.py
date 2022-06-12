from enum import Enum
import cv2 as cv

_GREEN_COLOR_RGB = (0, 255, 0)
_RED_COLOR_RGB = (0, 255, 0)
_KEYPOINT_RADIUS = 5


class EstimationResultType(Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


def draw_text(image, text, origin, font_scale, thickness):
    cv.putText(
        image,
        text,
        origin,
        cv.FONT_HERSHEY_SIMPLEX,
        font_scale,
        _GREEN_COLOR_RGB,
        thickness,
        cv.LINE_AA,
    )


def draw_circle(
    image, landmark_x, landmark_y, _type=EstimationResultType.SUCCESS.value
):
    if _type == EstimationResultType.SUCCESS.value:
        cv.circle(
            image, (landmark_x, landmark_y), _KEYPOINT_RADIUS, _GREEN_COLOR_RGB, 2
        )
    elif _type == EstimationResultType.FAILED.value:
        cv.circle(image, (landmark_x, landmark_y), _KEYPOINT_RADIUS, _RED_COLOR_RGB, 2)


def draw_line(image, landmark_x, landmark_y, _type=EstimationResultType.SUCCESS.value):
    if _type == EstimationResultType.SUCCESS.value:
        cv.line(image, landmark_x, landmark_y, _GREEN_COLOR_RGB, 2)
    elif _type == EstimationResultType.FAILED.value:
        cv.line(image, landmark_x, landmark_y, _RED_COLOR_RGB, 2)
