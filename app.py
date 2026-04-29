import streamlit as st
import cv2
import mediapipe as mp
from main import compute_ear, compute_mar, LEFT_EYE_IDX, RIGHT_EYE_IDX, MOUTH_IDX
import winsound
from math import isfinite

st.set_page_config(page_title="Drowsiness Detection", layout="wide")

st.title("Real-Time Drowsiness Detection")

ear_thresh = st.slider("EAR Threshold", 0.50, 0.90, 0.75, 0.01)
consec_frames = st.slider("Consecutive Frames (EAR)", 5, 40, 20, 1)
mar_thresh = st.slider("MAR Threshold", 0.40, 1.20, 0.70, 0.01)

start_button = st.button("Start Detection")
frame_placeholder = st.empty()
status_placeholder = st.empty()

if start_button:
    mp_face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(0)

    EAR_THRESH = ear_thresh
    CONSEC_FRAMES = consec_frames
    MAR_THRESH = mar_thresh
    YAWN_FRAMES = 10

    counter = 0
    yawn_counter = 0
    alarm_on = False

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            rgb.flags.writeable = True
            h, w, _ = frame.shape

            ear, mar = 0.0, 0.0
            status_text = "No face"
            color = (0, 0, 255)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                left_eye_pts, right_eye_pts, mouth_pts = [], [], []

                for idx in LEFT_EYE_IDX:
                    lm = face_landmarks.landmark[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    left_eye_pts.append((x, y))
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                for idx in RIGHT_EYE_IDX:
                    lm = face_landmarks.landmark[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    right_eye_pts.append((x, y))
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

                for idx in MOUTH_IDX:
                    lm = face_landmarks.landmark[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    mouth_pts.append((x, y))
                    cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

                if len(left_eye_pts) == 6 and len(right_eye_pts) == 6:
                    left_ear = compute_ear(left_eye_pts)
                    right_ear = compute_ear(right_eye_pts)
                    ear = (left_ear + right_ear) / 2.0

                    if len(mouth_pts) == 6:
                        mar = compute_mar(mouth_pts)

                    # yawn rule (simple version)
                    if mar > MAR_THRESH:
                        yawn_counter += 1
                    else:
                        yawn_counter = 0

                    if ear < EAR_THRESH:
                        counter += 1
                    else:
                        counter = 0
                        alarm_on = False

                    ear_drowsy = counter >= CONSEC_FRAMES
                    yawn_drowsy = yawn_counter >= YAWN_FRAMES

                    if ear_drowsy or yawn_drowsy:
                        status_text = "DROWSY"
                        color = (0, 0, 255)
                        if not alarm_on:
                            alarm_on = True
                            winsound.Beep(2500, 1000)
                    else:
                        status_text = "ALERT"
                        color = (0, 255, 0)
                else:
                    status_text = "Eye landmarks missing"
                    color = (0, 0, 255)

            # HUD
            cv2.rectangle(frame, (5, 5), (320, 110), (0, 0, 0), 2)
            cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.3f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, status_text, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Show in Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")
            status_placeholder.write(
                f"Status: **{status_text}** | EAR: `{ear:.3f}` | MAR: `{mar:.3f}`"
            )

            if not st.session_state.get("run", True):
                break

        cap.release()


