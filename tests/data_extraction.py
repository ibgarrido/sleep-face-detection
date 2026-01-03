import cv2
import mediapipe as mp
import csv
import numpy as np  
import time

#  Mediapipe Tasks API imports
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIG ---
MODEL_PATH = 'face_landmarker.task'

## Eyes index topology
## https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py
## https://stackoverflow.com/questions/69858216/mediapipe-facemesh-vertices-mapping
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

#https://arxiv.org/pdf/2408.05836
def calculate_ear(landmarks, index, img_w, img_h):
    coords = []
    for i in index:
        lm = landmarks[i]
        x, y = int(lm.x * img_w), int(lm.y * img_h)
        coords.append((x, y))
    v1 = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
    v2 = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
    h = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
    ear = (v1 + v2) / (2.0 * h)
    return ear

#detector config

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

detector = vision.FaceLandmarker.create_from_options(options)

#--- DATA COLLECTION ---
csv_file = 'eye_data.csv'
header = ['timestamp', 'left_ear', 'right_ear', 'avg_ear', 'label']

print("What class are you going to record now?")
print("0: Awake")
print("1: Sleepy")
class_id = input("Enter the number (0 or 1): ")

with open(csv_file, mode='a', newline='') as f:
    writer = csv.writer(f)
    if f.tell() == 0:
        writer.writerow(header)

    cap = cv2.VideoCapture(0)
    print("Press 'q' to stop.")

    #while video is opened
    while cap.isOpened():
        ret, frame = cap.read() #ret: bool, frame: image
        if not ret:
            break

        # MediaPipe Tasks require an mp.Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # detector 
        detection_result = detector.detect_for_video(
        mp_image, int(time.time() * 1000)
        )
        # Process results
        if detection_result.face_landmarks:
            # face_landmarks is a list of lists (one per face)
            face_landmarks = detection_result.face_landmarks[0] #only first face
            
            h, w, _ = frame.shape
            left_ear = calculate_ear(face_landmarks, LEFT_EYE_IDX, w, h)
            right_ear = calculate_ear(face_landmarks, RIGHT_EYE_IDX, w, h)
            avg_ear = (left_ear + right_ear) / 2.0

            writer.writerow([time.time(), left_ear, right_ear, avg_ear, class_id])

            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw points for visual reference
            for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
                pt = face_landmarks[idx]
                cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 2, (0,0,255), -1)

        cv2.imshow('Collector (Tasks API)', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()