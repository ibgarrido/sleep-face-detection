import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
from sklearn.preprocessing import StandardScaler

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

def process_video():
    logreg = joblib.load("sleep_model.pkl")

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print("Error: No se pudo abrir el video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    # <<< CAMBIO >>> buffers para features
    ear_raw_buffer = deque(maxlen=5)    # smoothing
    ear_feat_buffer = deque(maxlen=10)  # std + diff

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        frame_timestamp_ms = int((frame_count / fps) * 1000)
        detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

        status = "DESPIERTO"
        color = (0, 255, 0)

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]

            left = calculate_ear(landmarks, LEFT_EYE, width, height)
            right = calculate_ear(landmarks, RIGHT_EYE, width, height)
            avg_raw = (left + right) / 2.0

            #features
            ear_raw_buffer.append(avg_raw)
            avg_ear_smooth = np.mean(ear_raw_buffer)

            ear_feat_buffer.append(avg_ear_smooth)

            diff_ear_smooth = abs(ear_feat_buffer[-1] - ear_feat_buffer[-2]) if len(ear_feat_buffer) > 1 else 0.0
            ear_std = np.std(ear_feat_buffer) if len(ear_feat_buffer) >= 10 else 0.0

           
            X_frame = np.array([[avg_ear_smooth, diff_ear_smooth, ear_std]])
            p_sleep = logreg.predict_proba(X_frame)[0,1]

            if p_sleep >= SLEEP_THRESHOLD:
                status = "DORMIDO !!!"
                color = (0, 0, 255)

            
            cv2.putText(frame, f"Estado: {status}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            cv2.putText(frame, f"p_sleep: {p_sleep:.3f}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"EAR_s: {avg_ear_smooth:.3f}", (50, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
            cv2.putText(frame, f"diff: {diff_ear_smooth:.3f} std: {ear_std:.3f}", (50, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        out.write(frame)

        if frame_count % 30 == 0:
            print(f"Frame {frame_count} | p_sleep={p_sleep:.3f} | {status}")

    cap.release()
    out.release()
    print(f"Video guardado en {OUTPUT_VIDEO}")
    print(f"Tiempo total: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    process_video()
