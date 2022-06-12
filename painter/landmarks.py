from painter.draw import draw_text, draw_circle, draw_line
from pose.landmarks import get_landmark_key_points

_VISIBILITY_TOLERANCE = 0.5


def draw_landmarks(image, raw_landmarks):
    landmark_key_points = get_landmark_key_points(image, raw_landmarks)

    for key, landmark_keypoint in landmark_key_points.items():
        if landmark_keypoint.visibility < _VISIBILITY_TOLERANCE:
            continue
        draw_circle(image, landmark_keypoint.x, landmark_keypoint.y)
        draw_text(image, landmark_keypoint.x, landmark_keypoint.y, landmark_keypoint.z)

    if landmark_key_points:
        if (
            landmark_key_points["RIGHT_EYE_INNER"].visibility > _VISIBILITY_TOLERANCE
            and landmark_key_points["RIGHT_EYE"].visibility > _VISIBILITY_TOLERANCE
        ):
            draw_line(
                image,
                landmark_key_points["RIGHT_EYE_INNER"].point2d,
                landmark_key_points["RIGHT_EYE"].point2d,
            )
        if (
            landmark_key_points["RIGHT_EYE"].visibility > _VISIBILITY_TOLERANCE
            and landmark_key_points["RIGHT_EYE_OUTER"].visibility
            > _VISIBILITY_TOLERANCE
        ):
            draw_line(
                image,
                landmark_key_points["RIGHT_EYE"].point2d,
                landmark_key_points["RIGHT_EYE_OUTER"].point2d,
            )
        if (
            landmark_key_points["LEFT_EYE_INNER"].visibility > _VISIBILITY_TOLERANCE
            and landmark_key_points["LEFT_EYE"].visibility > _VISIBILITY_TOLERANCE
        ):
            draw_line(
                image,
                landmark_key_points["LEFT_EYE_INNER"].point2d,
                landmark_key_points["LEFT_EYE"].point2d,
            )
        if (
            landmark_key_points["LEFT_EYE"].visibility > _VISIBILITY_TOLERANCE
            and landmark_key_points["LEFT_EYE_OUTER"].visibility > _VISIBILITY_TOLERANCE
        ):
            draw_line(
                image,
                landmark_key_points["LEFT_EYE"].point2d,
                landmark_key_points["LEFT_EYE_OUTER"].point2d,
            )

        if (
            landmark_key_points["MOUTH_RIGHT"].visibility > _VISIBILITY_TOLERANCE
            and landmark_key_points["MOUTH_LEFT"].visibility > _VISIBILITY_TOLERANCE
        ):
            draw_line(
                image,
                landmark_key_points["MOUTH_RIGHT"].point2d,
                landmark_key_points["MOUTH_LEFT"].point2d,
            )
        if (
            landmark_key_points["RIGHT_SHOULDER"].visibility > _VISIBILITY_TOLERANCE
            and landmark_key_points["LEFT_SHOULDER"].visibility > _VISIBILITY_TOLERANCE
        ):
            draw_line(
                image,
                landmark_key_points["RIGHT_SHOULDER"].point2d,
                landmark_key_points["LEFT_SHOULDER"].point2d,
            )
        if (
            landmark_key_points["RIGHT_SHOULDER"].visibility > _VISIBILITY_TOLERANCE
            and landmark_key_points["RIGHT_ELBOW"].visibility > _VISIBILITY_TOLERANCE
        ):
            draw_line(
                image,
                landmark_key_points["RIGHT_SHOULDER"].point2d,
                landmark_key_points["RIGHT_ELBOW"].point2d,
            )
        if (
            landmark_key_points["RIGHT_ELBOW"].visibility > _VISIBILITY_TOLERANCE
            and landmark_key_points["RIGHT_WRIST"].visibility > _VISIBILITY_TOLERANCE
        ):
            draw_line(
                image,
                landmark_key_points["RIGHT_ELBOW"].point2d,
                landmark_key_points["RIGHT_WRIST"].point2d,
            )
        if (
            landmark_key_points["LEFT_SHOULDER"].visibility > _VISIBILITY_TOLERANCE
            and landmark_key_points["LEFT_ELBOW"].visibility > _VISIBILITY_TOLERANCE
        ):
            draw_line(
                image,
                landmark_key_points["LEFT_SHOULDER"].point2d,
                landmark_key_points["LEFT_ELBOW"].point2d,
            )
        if (
            landmark_key_points["LEFT_ELBOW"].visibility > _VISIBILITY_TOLERANCE
            and landmark_key_points["LEFT_WRIST"].visibility > _VISIBILITY_TOLERANCE
        ):
            draw_line(
                image,
                landmark_key_points["LEFT_ELBOW"].point2d,
                landmark_key_points["LEFT_WRIST"].point2d,
            )
        if (
            landmark_key_points["RIGHT_WRIST"].visibility > _VISIBILITY_TOLERANCE
            and landmark_key_points["RIGHT_PINKY_KNUCKLE"].visibility
            > _VISIBILITY_TOLERANCE
        ):
            draw_line(
                image,
                landmark_key_points["RIGHT_WRIST"].point2d,
                landmark_key_points["RIGHT_PINKY_KNUCKLE"].point2d,
            )
        if (
            landmark_key_points["RIGHT_PINKY_KNUCKLE"].visibility
            > _VISIBILITY_TOLERANCE
            and landmark_key_points["RIGHT_INDEX_KNUCKLE"].visibility
            > _VISIBILITY_TOLERANCE
        ):
            draw_line(
                image,
                landmark_key_points["RIGHT_PINKY_KNUCKLE"].point2d,
                landmark_key_points["RIGHT_INDEX_KNUCKLE"].point2d,
            )
        if (
            landmark_key_points["RIGHT_INDEX_KNUCKLE"].visibility
            > _VISIBILITY_TOLERANCE
            and landmark_key_points["RIGHT_THUMB_KNUCKLE"].visibility
            > _VISIBILITY_TOLERANCE
        ):
            draw_line(
                image,
                landmark_key_points["RIGHT_INDEX_KNUCKLE"].point2d,
                landmark_key_points["RIGHT_THUMB_KNUCKLE"].point2d,
            )
        if (
            landmark_key_points["RIGHT_THUMB_KNUCKLE"].visibility
            > _VISIBILITY_TOLERANCE
            and landmark_key_points["RIGHT_WRIST"].visibility > _VISIBILITY_TOLERANCE
        ):
            draw_line(
                image,
                landmark_key_points["RIGHT_THUMB_KNUCKLE"].point2d,
                landmark_key_points["RIGHT_WRIST"].point2d,
            )
        if (
            landmark_key_points["LEFT_WRIST"].visibility > _VISIBILITY_TOLERANCE
            and landmark_key_points["LEFT_PINKY_KNUCKLE"].visibility
            > _VISIBILITY_TOLERANCE
        ):
            draw_line(
                image,
                landmark_key_points["LEFT_WRIST"].point2d,
                landmark_key_points["LEFT_PINKY_KNUCKLE"].point2d,
            )
        if (
            landmark_key_points["LEFT_PINKY_KNUCKLE"].visibility > _VISIBILITY_TOLERANCE
            and landmark_key_points["LEFT_INDEX_KNUCKLE"].visibility
            > _VISIBILITY_TOLERANCE
        ):
            draw_line(
                image,
                landmark_key_points["LEFT_PINKY_KNUCKLE"].point2d,
                landmark_key_points["LEFT_INDEX_KNUCKLE"].point2d,
            )
        if (
            landmark_key_points["LEFT_INDEX_KNUCKLE"].visibility > _VISIBILITY_TOLERANCE
            and landmark_key_points["LEFT_THUMB_KNUCKLE"].visibility
            > _VISIBILITY_TOLERANCE
        ):
            draw_line(
                image,
                landmark_key_points["LEFT_INDEX_KNUCKLE"].point2d,
                landmark_key_points["LEFT_THUMB_KNUCKLE"].point2d,
            )
        if (
            landmark_key_points["LEFT_THUMB_KNUCKLE"].visibility > _VISIBILITY_TOLERANCE
            and landmark_key_points["LEFT_WRIST"].visibility > _VISIBILITY_TOLERANCE
        ):
            draw_line(
                image,
                landmark_key_points["LEFT_THUMB_KNUCKLE"].point2d,
                landmark_key_points["LEFT_WRIST"].point2d,
            )
        if (
            landmark_key_points["RIGHT_SHOULDER"].visibility > _VISIBILITY_TOLERANCE
            and landmark_key_points["RIGHT_HIP"].visibility > _VISIBILITY_TOLERANCE
        ):
            draw_line(
                image,
                landmark_key_points["RIGHT_SHOULDER"].point2d,
                landmark_key_points["RIGHT_HIP"].point2d,
            )
        if (
            landmark_key_points["LEFT_SHOULDER"].visibility > _VISIBILITY_TOLERANCE
            and landmark_key_points["LEFT_HIP"].visibility > _VISIBILITY_TOLERANCE
        ):
            draw_line(
                image,
                landmark_key_points["LEFT_SHOULDER"].point2d,
                landmark_key_points["LEFT_HIP"].point2d,
            )
        if (
            landmark_key_points["RIGHT_HIP"].visibility > _VISIBILITY_TOLERANCE
            and landmark_key_points["LEFT_HIP"].visibility > _VISIBILITY_TOLERANCE
        ):
            draw_line(
                image,
                landmark_key_points["RIGHT_HIP"].point2d,
                landmark_key_points["LEFT_HIP"].point2d,
            )

        if len(landmark_key_points) > 25:
            if (
                landmark_key_points["RIGHT_HIP"].visibility > _VISIBILITY_TOLERANCE
                and landmark_key_points["RIGHT_KNEE"].visibility > _VISIBILITY_TOLERANCE
            ):
                draw_line(
                    image,
                    landmark_key_points["RIGHT_HIP"].point2d,
                    landmark_key_points["RIGHT_KNEE"].point2d,
                )
            if (
                landmark_key_points["RIGHT_KNEE"].visibility > _VISIBILITY_TOLERANCE
                and landmark_key_points["RIGHT_ANKLE"].visibility
                > _VISIBILITY_TOLERANCE
            ):
                draw_line(
                    image,
                    landmark_key_points["RIGHT_KNEE"].point2d,
                    landmark_key_points["RIGHT_ANKLE"].point2d,
                )
            if (
                landmark_key_points["RIGHT_ANKLE"].visibility > _VISIBILITY_TOLERANCE
                and landmark_key_points["RIGHT_HEEL"].visibility > _VISIBILITY_TOLERANCE
            ):
                draw_line(
                    image,
                    landmark_key_points["RIGHT_ANKLE"].point2d,
                    landmark_key_points["RIGHT_HEEL"].point2d,
                )
            if (
                landmark_key_points["RIGHT_HEEL"].visibility > _VISIBILITY_TOLERANCE
                and landmark_key_points["RIGHT_FOOT_INDEX"].visibility
                > _VISIBILITY_TOLERANCE
            ):
                draw_line(
                    image,
                    landmark_key_points["RIGHT_HEEL"].point2d,
                    landmark_key_points["RIGHT_FOOT_INDEX"].point2d,
                )
            if (
                landmark_key_points["LEFT_HIP"].visibility > _VISIBILITY_TOLERANCE
                and landmark_key_points["LEFT_KNEE"].visibility > _VISIBILITY_TOLERANCE
            ):
                draw_line(
                    image,
                    landmark_key_points["LEFT_HIP"].point2d,
                    landmark_key_points["LEFT_KNEE"].point2d,
                )
            if (
                landmark_key_points["LEFT_KNEE"].visibility > _VISIBILITY_TOLERANCE
                and landmark_key_points["LEFT_ANKLE"].visibility > _VISIBILITY_TOLERANCE
            ):
                draw_line(
                    image,
                    landmark_key_points["LEFT_KNEE"].point2d,
                    landmark_key_points["LEFT_ANKLE"].point2d,
                )
            if (
                landmark_key_points["LEFT_ANKLE"].visibility > _VISIBILITY_TOLERANCE
                and landmark_key_points["LEFT_HEEL"].visibility > _VISIBILITY_TOLERANCE
            ):
                draw_line(
                    image,
                    landmark_key_points["LEFT_ANKLE"].point2d,
                    landmark_key_points["LEFT_HEEL"].point2d,
                )
            if (
                landmark_key_points["LEFT_HEEL"].visibility > _VISIBILITY_TOLERANCE
                and landmark_key_points["LEFT_FOOT_INDEX"].visibility
                > _VISIBILITY_TOLERANCE
            ):
                draw_line(
                    image,
                    landmark_key_points["LEFT_HEEL"].point2d,
                    landmark_key_points["LEFT_FOOT_INDEX"].point2d,
                )
    return image