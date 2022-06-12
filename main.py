import copy
import argparse
import cv2 as cv
import mediapipe as mp

from calc.cvfpscalc import CvFpsCalc
from calc.utils import calc_bounding_rect
from painter.draw import draw_bounding_rect
from painter.landmarks import draw_landmarks


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
    parser.add_argument("--use_brect", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    use_brect = args.use_brect

    # cap = cv.VideoCapture(cap_device)
    cap = cv.VideoCapture("samples/videos/pushup.mp4")

    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    pose = mp.solutions.pose.Pose(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    while True:
        display_fps = cvFpsCalc.get()
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = pose.process(image)
        # feedback = get_result(results)
        # if feedback:
        #     cv.putText(
        #         debug_image,
        #         str(feedback),
        #         (10, 60),
        #         cv.FONT_HERSHEY_SIMPLEX,
        #         1.0,
        #         (0, 0, 255),
        #         2,
        #         cv.LINE_AA,
        #     )
        if results.pose_landmarks is not None:
            bounding_rectangle_score = calc_bounding_rect(
                debug_image, results.pose_landmarks
            )
            debug_image = draw_landmarks(debug_image, results.pose_landmarks)
            debug_image = draw_bounding_rect(
                use_brect, debug_image, bounding_rectangle_score
            )
        cv.putText(
            debug_image,
            "FPS:" + str(display_fps),
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv.LINE_AA,
        )
        key = cv.waitKey(27)
        if key == 27:
            break
        cv.imshow("Posemvp Demo", debug_image)
    cap.release()
    cv.waitKey(0)
    cv.destroyAllWindows()
