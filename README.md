# sleep face detection

A real-time drownsiness detection using MediaPipe Face Landmarker, OpenCV and EAR (Eye aspect ratio).

This system detect wheter a person is awake or sleep ba using threshold value based on EAR in real time.


# How it works
1. Detects the face and extracts facial landmarks.
2. Computes vertical and horizontal eye distances.
3. Calculates the average EAR.
4. Calibrates the open-eye EAR during the first seconds.
5. Sets a personalized sleep threshold.
6. If EAR drops below threshold → state = DROWSY.


# requirements

- Python 3.10.19

# libraries

```
pip install -r requirements.txt
```
# Task file for face landmarker

Download the official model from MediaPipe:
https://developers.google.com/mediapipe/solutions/vision/face_landmarker

Save it as:
```
face_landmarker.task
```

# files

| File                          | Description                     |
| ----------------------------- | ------------------------------- |
| `sleep_detector_threshold.py` | Main script                     |
| `face_landmarker.task`        | MediaPipe Face Landmarker model |
| `README.md`                   | This file                       |


# Disclaimer

Works best with good frontal lighting.
Not a medical device.
Intended for experimental / academic use.

(Better implementations should use predicted models instead of a simple threshold)

