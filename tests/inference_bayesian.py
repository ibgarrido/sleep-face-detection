import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal


## Training class

class BayesianSleepPredictor:
    def __init__(self, model_path="bayes_model.pkl"):
        mu0, Sigma0, mu1, Sigma1, P, scaler = joblib.load(model_path)

        self.mu = [mu0, mu1]
        self.Sigma = [Sigma0, Sigma1]
        self.P = P
        self.scaler = scaler

        self.belief = np.array([0.99, 0.01])  # prior despierto/dormido

    def update(self, x_raw):
        x = self.scaler.transform([x_raw])[0]

        likelihood = np.array([
            multivariate_normal.pdf(x, self.mu[0], self.Sigma[0]),
            multivariate_normal.pdf(x, self.mu[1], self.Sigma[1])
        ])

        pred = self.P.T @ self.belief
        post = likelihood * pred
        self.belief = post / post.sum()

        return self.belief[1]  # prob dormir

# --- CONFIGURACIÓN ---
INPUT_VIDEO = '/data/input.mp4'
OUTPUT_VIDEO = '/data/output.mp4'
MODEL_PATH = 'face_landmarker.task'
SLEEP_THRESHOLD = 0.1

# Configuración MediaPipe
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def calculate_ear(landmarks, indices, w, h):
    coords = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices])
    v1 = np.linalg.norm(coords[1] - coords[5])
    v2 = np.linalg.norm(coords[2] - coords[4])
    hor = np.linalg.norm(coords[0] - coords[3])
    return (v1 + v2) / (2.0 * hor)

def compute_features(ear_buffer):
    avg = np.mean(ear_buffer)
    diff = ear_buffer[-1] - ear_buffer[-2] if len(ear_buffer) >= 2 else 0.0
    std = np.std(ear_buffer)
    return [avg, diff, std]

def process_video():
    predictor = BayesianSleepPredictor()

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print("Error: No se pudo abrir el video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(OUTPUT_VIDEO,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (width, height))

    ear_buffer = deque(maxlen=15)

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp = int((frame_count / fps) * 1000)

        result = detector.detect_for_video(mp_image, timestamp)

        status = "DESPIERTO"
        color = (0, 255, 0)

        if result.face_landmarks:
            lm = result.face_landmarks[0]
            left = calculate_ear(lm, LEFT_EYE, width, height)
            right = calculate_ear(lm, RIGHT_EYE, width, height)
            ear = (left + right) / 2

            ear_buffer.append(ear)

            if len(ear_buffer) >= 5:
                features = compute_features(ear_buffer)
                p_sleep = predictor.update(features)

                if p_sleep > 0.5:
                    status = "DORMIDO"
                    color = (0, 0, 255)
                else:
                    status = "DESPIERTO"
                    color = (0, 255, 0)

                cv2.putText(frame, f"P(dormido): {p_sleep:.2f}", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.putText(frame, f"Estado: {status}", (50, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)

        out.write(frame)

        if frame_count % 30 == 0:
            print(f"Frame {frame_count} | p_sleep={p_sleep:.3f} | Estado={status}")


    cap.release()
    out.release()
    print("Listo!")
