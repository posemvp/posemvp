import cv2 as cv


def draw_landmarks(image, landmarks, visibility_th=0.5):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        landmark_point.append([landmark.visibility, (landmark_x, landmark_y)])
        if landmark.visibility < visibility_th:
            continue
        if index == 0:
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 1:
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 2:
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 3:
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 4:
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 5:
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 6:
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 7:
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 8:
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 9:
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 10:
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 11:
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 12:
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 13:
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 14:
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 15:
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 16:
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 17:
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 18:
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 19:
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 20:
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 21:
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 22:
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 23:
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 24:
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 25:
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 26:
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 27:
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 28:
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 29:
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 30:
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 31:
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 32:
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)

        cv.putText(
            image,
            "z:" + str(round(landmark_z, 3)),
            (landmark_x - 10, landmark_y - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv.LINE_AA,
        )

    if len(landmark_point) > 0:
        if (
            landmark_point[1][0] > visibility_th
            and landmark_point[2][0] > visibility_th
        ):
            cv.line(image, landmark_point[1][1], landmark_point[2][1], (0, 255, 0), 2)
        if (
            landmark_point[2][0] > visibility_th
            and landmark_point[3][0] > visibility_th
        ):
            cv.line(image, landmark_point[2][1], landmark_point[3][1], (0, 255, 0), 2)

        if (
            landmark_point[4][0] > visibility_th
            and landmark_point[5][0] > visibility_th
        ):
            cv.line(image, landmark_point[4][1], landmark_point[5][1], (0, 255, 0), 2)
        if (
            landmark_point[5][0] > visibility_th
            and landmark_point[6][0] > visibility_th
        ):
            cv.line(image, landmark_point[5][1], landmark_point[6][1], (0, 255, 0), 2)

        if (
            landmark_point[9][0] > visibility_th
            and landmark_point[10][0] > visibility_th
        ):
            cv.line(image, landmark_point[9][1], landmark_point[10][1], (0, 255, 0), 2)
        if (
            landmark_point[11][0] > visibility_th
            and landmark_point[12][0] > visibility_th
        ):
            cv.line(image, landmark_point[11][1], landmark_point[12][1], (0, 255, 0), 2)
        if (
            landmark_point[11][0] > visibility_th
            and landmark_point[13][0] > visibility_th
        ):
            cv.line(image, landmark_point[11][1], landmark_point[13][1], (0, 255, 0), 2)
        if (
            landmark_point[13][0] > visibility_th
            and landmark_point[15][0] > visibility_th
        ):
            cv.line(image, landmark_point[13][1], landmark_point[15][1], (0, 255, 0), 2)
        if (
            landmark_point[12][0] > visibility_th
            and landmark_point[14][0] > visibility_th
        ):
            cv.line(image, landmark_point[12][1], landmark_point[14][1], (0, 255, 0), 2)
        if (
            landmark_point[14][0] > visibility_th
            and landmark_point[16][0] > visibility_th
        ):
            cv.line(image, landmark_point[14][1], landmark_point[16][1], (0, 255, 0), 2)
        if (
            landmark_point[15][0] > visibility_th
            and landmark_point[17][0] > visibility_th
        ):
            cv.line(image, landmark_point[15][1], landmark_point[17][1], (0, 255, 0), 2)
        if (
            landmark_point[17][0] > visibility_th
            and landmark_point[19][0] > visibility_th
        ):
            cv.line(image, landmark_point[17][1], landmark_point[19][1], (0, 255, 0), 2)
        if (
            landmark_point[19][0] > visibility_th
            and landmark_point[21][0] > visibility_th
        ):
            cv.line(image, landmark_point[19][1], landmark_point[21][1], (0, 255, 0), 2)
        if (
            landmark_point[21][0] > visibility_th
            and landmark_point[15][0] > visibility_th
        ):
            cv.line(image, landmark_point[21][1], landmark_point[15][1], (0, 255, 0), 2)
        if (
            landmark_point[16][0] > visibility_th
            and landmark_point[18][0] > visibility_th
        ):
            cv.line(image, landmark_point[16][1], landmark_point[18][1], (0, 255, 0), 2)
        if (
            landmark_point[18][0] > visibility_th
            and landmark_point[20][0] > visibility_th
        ):
            cv.line(image, landmark_point[18][1], landmark_point[20][1], (0, 255, 0), 2)
        if (
            landmark_point[20][0] > visibility_th
            and landmark_point[22][0] > visibility_th
        ):
            cv.line(image, landmark_point[20][1], landmark_point[22][1], (0, 255, 0), 2)
        if (
            landmark_point[22][0] > visibility_th
            and landmark_point[16][0] > visibility_th
        ):
            cv.line(image, landmark_point[22][1], landmark_point[16][1], (0, 255, 0), 2)
        if (
            landmark_point[11][0] > visibility_th
            and landmark_point[23][0] > visibility_th
        ):
            cv.line(image, landmark_point[11][1], landmark_point[23][1], (0, 255, 0), 2)
        if (
            landmark_point[12][0] > visibility_th
            and landmark_point[24][0] > visibility_th
        ):
            cv.line(image, landmark_point[12][1], landmark_point[24][1], (0, 255, 0), 2)
        if (
            landmark_point[23][0] > visibility_th
            and landmark_point[24][0] > visibility_th
        ):
            cv.line(image, landmark_point[23][1], landmark_point[24][1], (0, 255, 0), 2)

        if len(landmark_point) > 25:
            if (
                landmark_point[23][0] > visibility_th
                and landmark_point[25][0] > visibility_th
            ):
                cv.line(
                    image, landmark_point[23][1], landmark_point[25][1], (0, 255, 0), 2
                )
            if (
                landmark_point[25][0] > visibility_th
                and landmark_point[27][0] > visibility_th
            ):
                cv.line(
                    image, landmark_point[25][1], landmark_point[27][1], (0, 255, 0), 2
                )
            if (
                landmark_point[27][0] > visibility_th
                and landmark_point[29][0] > visibility_th
            ):
                cv.line(
                    image, landmark_point[27][1], landmark_point[29][1], (0, 255, 0), 2
                )
            if (
                landmark_point[29][0] > visibility_th
                and landmark_point[31][0] > visibility_th
            ):
                cv.line(
                    image, landmark_point[29][1], landmark_point[31][1], (0, 255, 0), 2
                )
            if (
                landmark_point[24][0] > visibility_th
                and landmark_point[26][0] > visibility_th
            ):
                cv.line(
                    image, landmark_point[24][1], landmark_point[26][1], (0, 255, 0), 2
                )
            if (
                landmark_point[26][0] > visibility_th
                and landmark_point[28][0] > visibility_th
            ):
                cv.line(
                    image, landmark_point[26][1], landmark_point[28][1], (0, 255, 0), 2
                )
            if (
                landmark_point[28][0] > visibility_th
                and landmark_point[30][0] > visibility_th
            ):
                cv.line(
                    image, landmark_point[28][1], landmark_point[30][1], (0, 255, 0), 2
                )
            if (
                landmark_point[30][0] > visibility_th
                and landmark_point[32][0] > visibility_th
            ):
                cv.line(
                    image, landmark_point[30][1], landmark_point[32][1], (0, 255, 0), 2
                )
    return image
