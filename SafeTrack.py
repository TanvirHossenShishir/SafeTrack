import cv2
import numpy as np
import timeit
import pygame
import itertools
import tensorflow as tf

GREEN = (0, 255, 0)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)
PINK = (204, 0, 204)
CYAN = (255, 255, 0)
GRAY = (128, 128, 128)
FONT = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)

model = tf.keras.models.load_model('model.h5')

eye_images = []
counter = 0
batch_size = 4

pygame.mixer.init()
sound = pygame.mixer.Sound('assets/alert.mp3')

modelFile = "assets/opencv_face_detector_uint8.pb"
configFile = "assets/opencv_face_detector.pbtxt"
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

frameno = 0
start_time = timeit.default_timer()
start_time2 = start_time
mxfps = 0
prediction = 0

def display_eye_status(frame):
    global counter
    global prediction
    x1, y1, w1, h1 = 0, 0, 20, 30
    if counter == 0:
        TEXT_COLOR = GREEN
    elif counter >= 15:
        TEXT_COLOR = RED
    else:
        TEXT_COLOR = ORANGE

    print("Prediction: ", prediction)

    if prediction >= 0.5:
        if counter > 50:
            couter = 30
        if counter > 30:
            counter -= 3
        if counter > 0:
            counter -= 1
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


def process_batch(batch):
    global counter
    global prediction
    predictions = model.predict(batch)
    prediction = 0
    for i, pred in enumerate(predictions):
        prediction += pred
        print(pred)

    prediction /= batch_size
        

def add_to_batch(eye_image):
    img_size = 64
    eye_array = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
    eye_array = cv2.resize(eye_array, (img_size, img_size))
    eye_array = np.expand_dims(eye_array, axis=-1)
    eye_array = (eye_array - np.min(eye_array)) / (np.max(eye_array) - np.min(eye_array))
    eye_images.append(eye_array)

    if len(eye_images) == batch_size:
        batch = np.array(eye_images)
        process_batch(batch)
        eye_images.clear()


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
            x = int(detections[0, 0, i, 3] * frameWidth)
            y = int(detections[0, 0, i, 4] * frameHeight)
            w = int(detections[0, 0, i, 5] * frameWidth) - x
            h = int(detections[0, 0, i, 6] * frameHeight) - y

            cv2.rectangle(frame, (x - 20, y - 10), (x + w + 20, y + h), PINK, 2)

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

            r_eye_roi = frame[y + ry: y + rh, x + rx: x + rw]
            l_eye_roi = frame[y + ly: y + lh, x + lx: x + lw]

            add_to_batch(r_eye_roi)
            add_to_batch(l_eye_roi)

    elapsed_time = timeit.default_timer() - start_time
    fps = int(1 / elapsed_time)
    if fps > mxfps:
        mxfps = fps

    x1, y1, w1, h1 = 0, 0, 20, 30

    cv2.putText(frame, "FPS: " + str(fps), (x1 + w1, y1 + h1 + 430), FONT, 0.6, GRAY, 1)
    cv2.putText(frame, "Max: " + str(mxfps), (x1 + w1 + 100, y1 + h1 + 430), FONT, 0.6, GRAY, 1)

    time_text = "Time: {:.2f}".format(timeit.default_timer() - start_time2)
    cv2.putText(frame, time_text, (x1 + w1 + 200, y1 + h1 + 430), FONT, 0.6, GRAY, 1)

    display_eye_status(frame)

    cv2.imshow("SafeTrack", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    start_time = timeit.default_timer()

cap.release()
cv2.destroyAllWindows()