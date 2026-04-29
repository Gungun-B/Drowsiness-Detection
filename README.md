# Drowsiness-Detection

A real-time **Drowsiness Detection System** that monitors a person's eye state and alerts them when signs of fatigue are detected. This project is designed to improve safety in scenarios like driving, studying, or operating machinery.


##  Features

*  Real-time eye tracking using computer vision
*  Detects drowsiness based on eye closure patterns
*  Instant alert system (alarm/sound notification)
*  Works with live webcam feed
*  Lightweight and efficient


##  Tech Stack

* **Programming Language:** Python
* **Libraries & Tools:**

  * OpenCV
  * Dlib / Mediapipe (depending on your implementation)
  * NumPy
  * Scipy (optional)


##  Usage

Run the main script:

```bash
python src/main.py
```

* The webcam will start automatically.
* If drowsiness is detected, an alert sound will trigger.


##  How It Works

* The system detects facial landmarks using a pre-trained model.
* It calculates the **Eye Aspect Ratio (EAR)** to determine eye openness.
* If EAR falls below a threshold for a sustained period, the system classifies the user as drowsy.
* An alert is triggered to wake the user.


##  Applications

* Driver safety systems 🚗
* Workplace fatigue monitoring 🏭
* Student alertness tracking 📚


##  Limitations

* Performance may vary in low lighting conditions
* Requires a clear view of the face
* Glasses or obstructions can affect accuracy


