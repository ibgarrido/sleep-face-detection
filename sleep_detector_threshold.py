import cv2
import mediapipe as mp
import numpy as np
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "face_landmark_API/face_landmarker.task"

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def calculate_ear(landmarks, index, img_w, img_h):
    coords = []
    for i in index:
        lm = landmarks[i]
        x, y = int(lm.x * img_w), int(lm.y * img_h)
        coords.append((x, y))
    v1 = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
    v2 = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
    h = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
    return (v1 + v2) / (2.0 * h)

# Detector
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)
detector = vision.FaceLandmarker.create_from_options(options)

# Calibration
CALIBRATION_TIME = 3
print("Calibrating... Please keep your eyes open.")

cap = cv2.VideoCapture(0)
ear_values = []
start_time = time.time()
timestamp_ms = 0

while time.time() - start_time < CALIBRATION_TIME:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w = frame.shape[:2]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(mp_image, timestamp_ms)
    timestamp_ms += 33

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]
        ear = (calculate_ear(landmarks, LEFT_EYE_IDX, img_w, img_h) +
               calculate_ear(landmarks, RIGHT_EYE_IDX, img_w, img_h)) / 2
        ear_values.append(ear)

if ear_values:
    open_ear_avg = np.mean(ear_values)
    sleep_threshold = open_ear_avg * 0.75
    print(f"Calibrated threshold: {sleep_threshold:.3f}")
else:
    sleep_threshold = 0.2
    print("Calibration failed — using default threshold.")

cap.release()

print("Detection started — press Q to quit.")

# Detection loop
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w = frame.shape[:2]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(mp_image, timestamp_ms)
    timestamp_ms += 33

    status = "AWAKE"
    color = (0, 255, 0)

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]
        ear = (calculate_ear(landmarks, LEFT_EYE_IDX, img_w, img_h) +
               calculate_ear(landmarks, RIGHT_EYE_IDX, img_w, img_h)) / 2

        if ear < sleep_threshold:
            status = "ASLEEP"
            color = (0, 0, 255)

        cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(frame, f"Status: {status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow("Sleep Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
