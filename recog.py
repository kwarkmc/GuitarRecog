import cv2 as cv
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

capture = cv.VideoCapture(0)

font = cv.FONT_HERSHEY_DUPLEX

with mp_hands.Hands (
    max_num_hands = 2,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
) as hands:

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            continue
        frame = cv.cvtColor(cv.flip(frame, 1), cv.COLOR_BGR2RGB)
        results = hands.process(frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                finger1 = hand_landmarks.landmark[8]

                """
                finger1_x = float("{0:0.3f}".format(finger1.x))
                finger1_y = float("{0:0.3f}".format(finger1.y))
                coordinate = "x:{0}, y:{1}".format(finger1_x, finger1_y)
                cv.putText(frame, coordinate, (15, 15), font, 1, (255, 0, 0))
                """

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv.imshow("Finger", frame)
        if cv.waitKey(1) == 27:
            break

capture.release()
cv.destroyAllWindows()