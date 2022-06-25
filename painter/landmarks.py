from painter.draw import draw_text, draw_circle, draw_line

_VISIBILITY_TOLERANCE = 0.5


def draw_landmarks(image, landmark_key_points):
    for key, landmark_keypoint in landmark_key_points.items():
        if landmark_keypoint.visibility < _VISIBILITY_TOLERANCE:
            continue
        draw_circle(image, landmark_keypoint.point2d[0], landmark_keypoint.point2d[1])
        draw_text(
            image,
            f"z:{round(landmark_keypoint.normalizedPoint3d.z, 3)}",
            (landmark_keypoint.point2d[0] - 10, landmark_keypoint.point2d[1] - 10),
            0.5,
            1,
        )

    if landmark_key_points:
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
            and landmark_key_points["RIGHT_ANKLE"].visibility > _VISIBILITY_TOLERANCE
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
