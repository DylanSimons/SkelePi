

import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
FACE_3D = np.array([
    (0.0, 0.0, 0.0),         # Nose tip
    (0.0, -63.6, -12.5),     # Chin
    (-43.3, 32.7, -26.0),    # Left eye left corner
    (43.3, 32.7, -26.0),     # Right eye right corner
    (-28.9, -28.9, -24.1),   # Left Mouth corner
    (28.9, -28.9, -24.1)     # Right mouth corner
])
LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        yaw = pitch = roll = 0.0
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            # Use the same logic as pose.js predictLoop for head pose estimation
            # Get required landmarks
            # 1: nose tip, 33: left eye outer, 263: right eye outer, 152: chin, 10: forehead
            idx_nose = 1
            idx_left_eye = 33
            idx_right_eye = 263
            idx_chin = 152
            idx_forehead = 10
            lm_nose = face_landmarks.landmark[idx_nose]
            lm_left_eye = face_landmarks.landmark[idx_left_eye]
            lm_right_eye = face_landmarks.landmark[idx_right_eye]
            lm_chin = face_landmarks.landmark[idx_chin]
            lm_forehead = face_landmarks.landmark[idx_forehead]

            # Convert to 3D coordinates (normalized, so use as-is)
            # For pitch, use y and z; for yaw, use x and z; for roll, use x and y
            dx = lm_right_eye.x - lm_left_eye.x
            dy = lm_right_eye.y - lm_left_eye.y
            dz = lm_right_eye.z - lm_left_eye.z
            yaw = np.degrees(np.arctan2(dx, dz))
            roll = np.degrees(np.arctan2(dy, dx))

            dy_pitch = lm_chin.y - lm_forehead.y
            dz_pitch = lm_chin.z - lm_forehead.z
            pitch = np.degrees(np.arctan2(dy_pitch, dz_pitch))

            #pitch and yaw are 90 degrees off for some reason
            pitch -= 90
            yaw -= 90

        print(f"Yaw: {yaw:.2f} | Pitch: {pitch:.2f} | Roll: {roll:.2f}")

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()