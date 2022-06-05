import mediapipe as mp
import cv2 as cv

mp_pose = mp.solutions.pose
upper_body_only = False
min_detection_confidence = 0.5
min_tracking_confidence = 0.5

pose = mp_pose.Pose(
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence,
)

image = cv.imread("images/mountain_pose.jpg")

results = pose.process(image)

image_pose_x = []
image_pose_y = []

for i in range(33):
    image_pose_x.append(results.pose_landmarks.landmark[i].x)
    image_pose_y.append(results.pose_landmarks.landmark[i].y)
