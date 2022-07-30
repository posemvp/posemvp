# Extracting points from our input

import cv2
import mediapipe as mp
import numpy as np
import time
from dtaidistance import dtw
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture("/videos/warrior_II_pose.mp4")

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # out = cv2.VideoWriter(
    #     "output-1.mp4", cv2.VideoWriter_fourcc("M", "P", "4", "2"), 24, (450, 650)
    # )
    j = 0
    Model_Points = np.zeros((165, 2, 22))
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            print(landmarks)
        except:
            pass

        # Normalizing the data and Storing data of the Model

        for i in range(11, 33):
            n1 = landmarks[i].x
            n2 = landmarks[i].y
            normalize_coordinate = [n1, n2]
            normalize_coordinate = normalize_coordinate / np.linalg.norm(
                normalize_coordinate
            )
            Model_Points[j][0][i - 11] = normalize_coordinate[0]
            Model_Points[j][1][i - 11] = normalize_coordinate[1]

        j = j + 1
        # Render detections
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )

        image = cv2.resize(image, (450, 650))

        # out.write(image)
        cv2.imshow("Pose Estimation for Test", image)

        if cv2.waitKey(5) & 0xFF == "c":
            break

    cap.release()
    # out.release()
    cv2.destroyAllWindows()

# Calculating the score by Comparing with a Given Model

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    cap = cv2.VideoCapture("video-1.mp4")
    # out = cv2.VideoWriter(
    #     "test-output-1.mp4", cv2.VideoWriter_fourcc("M", "P", "4", "2"), 24, (450, 650)
    # )
    j = 0
    Test_Points = np.zeros((149, 2, 22))
    percentage = np.zeros(165)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
        except:
            pass

        # Normalizing the data and Storing data of the Model

        for i in range(11, 33):
            n1 = landmarks[i].x
            n2 = landmarks[i].y
            normalize_coordinate = [n1, n2]
            normalize_coordinate = normalize_coordinate / np.linalg.norm(
                normalize_coordinate
            )
            Test_Points[j][0][i - 11] = normalize_coordinate[0]
            Test_Points[j][1][i - 11] = normalize_coordinate[1]

        # Calculating the Score using Dynamic Time Warping

        s1 = Test_Points[j][0]
        s2 = Model_Points[j][0]
        distanceS = dtw.distance(s1, s2)
        k1 = Test_Points[j][1]
        k2 = Model_Points[j][1]
        distanceK = dtw.distance(k1, k2)

        percentageS = 100 - (distanceS * 100)
        percentageK = 100 - (distanceK * 100)
        percentage[j] = (percentageS + percentageK) / 2

        j = j + 1

        # Render detections
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )

        image = cv2.resize(image, (450, 650))

        # out.write(image)
        cv2.imshow("Pose Estimation for the given model", image)

        if cv2.waitKey(5) & 0xFF == "c":
            break

    cap.release()
    # out.release()
    cv2.destroyAllWindows()
# Plotting the Score Data real time

plt.rcParams["animation.html"] = "jshtml"

fig = plt.figure()

ax = fig.add_subplot(111)
fig.show()

plt.xlabel("Epoch of Video")
plt.ylabel("Percentage of score at similar time instance")
plt.title("Score Calculation of Test Video and Model Video")

i = 0
x, y = [], []

while percentage[i] != 0:
    x.append(i)
    y.append(percentage[i])

    ax.plot(x, y, color="b")

    fig.canvas.draw()

    time.sleep(0.1)
    i += 1

plt.close()
