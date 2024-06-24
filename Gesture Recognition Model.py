#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
try:
    model = load_model(r'C:\Users\Shreya\Downloads\hand-gesture-recognition-code\mp_hand_gesture')
except Exception as e:
    print("Error loading model:", e)
    exit()

# Load class names
try:
    with open(r"C:\Users\Shreya\Downloads\hand-gesture-recognition-code\gesture.names", 'r') as f:
        classNames = f.read().split('\n')
except Exception as e:
    print("Error loading class names:", e)
    exit()

# Initialize the webcam for Hand Gesture Recognition Python project
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame")
        break
    
    frame = cv2.flip(frame, 1)
    x, y, _ = frame.shape

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    result = hands.process(framergb)

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
            
            # Predict gesture in Hand Gesture Recognition project
            prediction = model.predict([landmarks])
            classID = np.argmax(prediction)
            className = classNames[classID]

            # show the prediction on the frame
            cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()


# In[ ]:




