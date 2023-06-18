import cv2
import numpy as np
import tensorflow as tf
import pygame
import timeit
import itertools

start_time = timeit.default_timer()
start_time2 = start_time

counter = 0
frameno = 0
mx = 0

model = tf.keras.models.load_model('model.h5')

modelFile = "assets/opencv_face_detector_uint8.pb"
configFile = "assets/opencv_face_detector.pbtxt"
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

pygame.mixer.init()
audio_file = "assets/alert.mp3"
sound = pygame.mixer.Sound(audio_file)

GREEN = (0, 165, 0)
ORANGE = (0, 212, 255)
RED = (0, 0, 255)
GRAY = (165, 165, 165)
PINK = (96, 27, 216)
CYAN = (154, 166, 38)

FONT = cv2.FONT_HERSHEY_SIMPLEX

for i in itertools.count():
    frameno = frameno + 1
    print(frameno)

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameHeight, frameWidth = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)

            x = x1
            y = y1
            w = x2 - x1
            h = y2 - y1

            cv2.rectangle(frame, (x1 - 20, y1 - 10), (x2 + 20, y2), PINK, 2)

            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            rx = 0
            ry = int(h / 4)
            rw = int(w / 2)
            rh = int(h / 4) + int(h / 4)

            lx = w - int(w / 2)
            ly = int(h / 4)
            lw = w
            lh = int(h / 4) + int(h / 4)

            cv2.rectangle(frame, (x + rx, y + ry), (x + rw - 10, y + rh), CYAN, 2)
            cv2.rectangle(frame, (x + lx + 10, y + ly), (x + lw, y + lh), CYAN, 2)

            face_roi = frame[y1:y2, x1:x2]
            r_eye_roi = frame[y + ry: y + rh, x + rx: x + rw]
            l_eye_roi = frame[y + ly: y + lh, x + lx: x + lw]

            img_size = 64

            r_eye_array = cv2.cvtColor(r_eye_roi, cv2.COLOR_BGR2GRAY)
            r_eye_array = cv2.resize(r_eye_array, (img_size, img_size))
            r_eye_array = np.expand_dims(r_eye_array, axis=-1)
            r_eye_array = (np.array(r_eye_array) - np.min(r_eye_array)) / (np.max(r_eye_array) - np.min(r_eye_array))

            l_eye_array = cv2.cvtColor(l_eye_roi, cv2.COLOR_BGR2GRAY)
            l_eye_array = cv2.resize(l_eye_array, (img_size, img_size))
            l_eye_array = np.expand_dims(l_eye_array, axis=-1)
            l_eye_array = (np.array(l_eye_array) - np.min(l_eye_array)) / (np.max(l_eye_array) - np.min(l_eye_array))

            r_eye_prediction = model.predict(np.expand_dims(r_eye_array, 0))
            l_eye_prediction = model.predict(np.expand_dims(l_eye_array, 0))

            x1, y1, w1, h1 = 0, 0, 20, 30

            if counter == 0:
                TEXT_COLOR = GREEN
            elif counter >= 15:
                TEXT_COLOR = RED
            else:
                TEXT_COLOR = ORANGE

            if r_eye_prediction >= 0.5 or l_eye_prediction >= 0.5:
                counter = 0
                cv2.putText(frame, "Open Eyes", (x1 + w1, y1 + h1), FONT, 0.7, TEXT_COLOR, 2)
            else:
                counter = counter + 1
                cv2.putText(frame, "Closed Eyes", (x1 + w1, y1 + h1), FONT, 0.7, TEXT_COLOR, 2)

            if counter >= 15 and not pygame.mixer.get_busy():
                cv2.putText(frame, "Drowsy Alert!!", (x1 + w1, y1 + h1 + 40), FONT, 0.7, TEXT_COLOR, 2)
                sound.play()

            text = "Drowsiness: " + str(counter)
            text_size, _ = cv2.getTextSize(text, FONT, 0.7, 2)
            text_x = frameWidth - text_size[0] - 20
            text_y = y1 + h1
            cv2.putText(frame, text, (text_x, text_y), FONT, 0.7, TEXT_COLOR, 2)

    elapsed_time = timeit.default_timer() - start_time
    fps = int(1 / elapsed_time)
    if fps > mx:
        mx = fps

    cv2.putText(frame, "FPS: " + str(fps), (x1 + w1, y1 + h1 + 430), FONT, 0.6, GRAY, 1)

    cv2.putText(frame, "Max: " + str(mx), (x1 + w1 + 100, y1 + h1 + 430), FONT, 0.6, GRAY, 1)

    time_text = "Time: {:.2f}".format(timeit.default_timer() - start_time2)
    cv2.putText(frame, time_text, (x1 + w1 + 200, y1 + h1 + 430), FONT, 0.6, GRAY, 1)

    cv2.imshow("SafeTrack", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    start_time = timeit.default_timer()

cap.release()
cv2.destroyAllWindows()

