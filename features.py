import numpy as np
import cv2
import mediapipe

#https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
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

def calculate_mouth_aspect_ratio(landmarks, index, img_w, img_h):
    coords = []
    for i in index:
        lm = landmarks[i]
        x, y = int(lm.x * img_w), int(lm.y * img_h)
        coords.append((x, y))
    v1 = np.linalg.norm(np.array(coords[13]) - np.array(coords[19]))
    v2 = np.linalg.norm(np.array(coords[14]) - np.array(coords[18]))
    v3 = np.linalg.norm(np.array(coords[15]) - np.array(coords[17]))
    h = np.linalg.norm(np.array(coords[12]) - np.array(coords[16]))
    mar = (v1 + v2 + v3) / (3.0 * h)
    return mar




if __name__ == "__main__":
    print("This module provides functions to calculate facial features such as EAR and MAR.")

# snippet from tests/data_extraction.py
    coords.append((x, y))
    v1 = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
    v2 = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
    h = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
    ear = (v1 + v2) / (2.0 * h)
    return ear