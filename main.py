import cv2
import mediapipe as mp
import numpy as np
import math
import winsound
import csv
import time
from math import isfinite

log_file = open("ear_log.csv", mode="w", newline="")
csv_writer = csv.writer(log_file)
csv_writer.writerow(["timestamp", "ear", "mar", "status"])

LEFT_EYE_IDX = [33, 159, 158, 133, 153, 145]
RIGHT_EYE_IDX = [362, 380, 374, 263, 386, 385]

MOUTH_IDX = [61, 291, 13, 14, 308, 78]  # [P1, P4, P2, P6, P3, P5]

def euclidean_dist(a, b):
    return math.dist(a, b)  

def compute_ear(eye_pts):
    p1, p2, p3, p4, p5, p6 = eye_pts
    vert1 = euclidean_dist(p2, p6)
    vert2 = euclidean_dist(p3, p5)
    horiz = euclidean_dist(p1, p4)
    if horiz == 0:
        return 0.0
    ear = (vert1 + vert2) / (2.0 * horiz)
    return ear

def compute_mar(mouth_pts):
    p1, p4, p2, p6, p3, p5 = mouth_pts
    vert1 = euclidean_dist(p2, p6)
    vert2 = euclidean_dist(p3, p5)
    horiz = euclidean_dist(p1, p4)
    if horiz == 0:
        return 0.0
    mar = (vert1 + vert2) / (2.0 * horiz)
    return mar

# Setup MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Create FaceMesh object
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,       # gives more detailed eye/iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    EAR_THRESH = 0.25
    CONSEC_FRAMES = 25
    MAR_THRESH = 0.7      
    YAWN_FRAMES = 10      # frames mouth must stay open - seconds * fps (e.g. 10 frames at 30fps = ~0.33 seconds)
    yawn_counter = 0
    counter = 0
    alarm_on = False

    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Optional resize
        frame = cv2.resize(frame, (640, 480))

        # Convert BGR (OpenCV) -> RGB (MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # To improve performance
        rgb_frame.flags.writeable = False
        results = face_mesh.process(rgb_frame)
        
        rgb_frame.flags.writeable = True
        h, w, _ = frame.shape  # original BGR frame size
        mar = 0.0
        # Draw face mesh landmarks on the original BGR frame
        if results.multi_face_landmarks:
            # take first (or best) face
            face_landmarks = results.multi_face_landmarks[0]

            left_eye_pts = []
            right_eye_pts = []
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

            mouth_pts = []
            for idx in MOUTH_IDX:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                mouth_pts.append((x, y))
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)  # blue points on mouth

            if len(left_eye_pts) == 6 and len(right_eye_pts) == 6:
                left_ear = compute_ear(left_eye_pts)
                right_ear = compute_ear(right_eye_pts)
                ear = (left_ear + right_ear) / 2.0   

                # Distance compensation
                face_center = (int(face_landmarks.landmark[1].x * w), int(face_landmarks.landmark[1].y * h))
                distance_factor = max(0.5, min(1.5, face_center[0] / 320))
                ear_normalized = ear * distance_factor

                if len(mouth_pts) == 6:
                    mar = compute_mar(mouth_pts)
                
                if mar > MAR_THRESH:
                    yawn_counter += 1
                else:
                    yawn_counter = 0

                # Use normalized for logic
                if ear_normalized < EAR_THRESH:
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
                
                if isfinite(ear) and isfinite(mar):
                    csv_writer.writerow([time.time(), round(ear, 4), round(mar, 4), status_text])

                cv2.rectangle(frame, (5, 5), (320, 110), (0, 0, 0), 2)
                
                # Show both values
                cv2.putText(frame, f"EAR: {ear:.3f}({ear_normalized:.3f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.putText(frame, f"MAR: {mar:.3f}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.putText(frame, status_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                cv2.putText(frame, f"THRESH: {EAR_THRESH:.2f}  FRAMES: {CONSEC_FRAMES}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.rectangle(frame, (5, 5), (320, 110), (0, 0, 0), 2)
                
                cv2.putText(frame, "Eye landmarks missing", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.putText(frame, f"MAR: {mar:.3f}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.putText(frame, f"THRESH: {EAR_THRESH:.2f}  FRAMES: {CONSEC_FRAMES}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.rectangle(frame, (5, 5), (320, 110), (0, 0, 0), 2)
            
            cv2.putText(frame, "No face", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(frame, f"THRESH: {EAR_THRESH:.2f}  FRAMES: {CONSEC_FRAMES}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(frame, f"MAR: {mar:.3f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    log_file.close()

cap.release()
cv2.destroyAllWindows()
