import copy
import argparse
import cv2 as cv
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

from calc.cvfpscalc import CvFpsCalc
from painter.draw import draw_text
from painter.landmarks import draw_landmarks
from pose.compare import compare_pose
from pose.landmarks import get_landmark_key_points


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help="cap width", type=int, default=960)
    parser.add_argument("--height", help="cap height", type=int, default=540)
    parser.add_argument(
        "--min_detection_confidence",
        help="min_detection_confidence",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--min_tracking_confidence",
        help="min_tracking_confidence",
        type=int,
        default=0.5,
    )
    return parser.parse_args()


def _get_image(pose_name):
    return cv.imread(f"exercises_detector/yoga/pose_images/{pose_name}.jpg")


def _plot_image_pose_graph(name, landmarks):
    image_vector_x = []
    image_vector_y = []

    for i, key_point in landmarks.items():
        image_vector_x.append(key_point.to_dict()["normalized_point3d"]["x"])
        image_vector_y.append(key_point.to_dict()["normalized_point3d"]["y"])

    image_x_points = np.array(image_vector_x)
    image_y_points = np.array(image_vector_y)

    plt.plot(image_x_points, image_y_points)
    plt.savefig(f"samples/graphs/{name}.png")


if __name__ == "__main__":
    args = get_args()

    # cap = cv.VideoCapture(args.device)
    cap = cv.VideoCapture("samples/videos/warrior_II_pose.mp4")

    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    pose = mp.solutions.pose.Pose(
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    image_name = "warrior_II_pose"
    image = _get_image(image_name)
    pose_landmarks = pose.process(image).pose_landmarks
    image_key_points = get_landmark_key_points(image, pose_landmarks)
    # _plot_image_pose_graph(image_name, image_key_points)

    cvFpsCalc = CvFpsCalc(buffer_len=10)

    while True:
        display_fps = cvFpsCalc.get()
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        pose_landmarks = pose.process(image).pose_landmarks
        if pose_landmarks is not None:
            landmark_key_points = get_landmark_key_points(image, pose_landmarks)
            compare_pose(image_key_points, landmark_key_points)
            debug_image = draw_landmarks(debug_image, landmark_key_points)
        draw_text(debug_image, "FPS:" + str(display_fps), (10, 30), 1.0, 2)
        key = cv.waitKey(27)
        if key == 27:
            break
        cv.imshow("Posemvp Demo", debug_image)
    plt.show()
    cap.release()
    cv.waitKey(0)
    cv.destroyAllWindows()
