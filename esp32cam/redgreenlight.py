import mediapipe as mp
import cv2
import numpy as np
import time
from playsound import playsound
import urllib.request

# Initialize variables
cap = cv2.VideoCapture(1)
cPos = 0
startT = 0
endT = 0
userSum = 0
dur = 0
isAlive = 1
isInit = False
cStart, cEnd = 0, 0
isCinit = False
tempSum = 0
winner = 0
inFrame = 0
inFramecheck = False
thresh = 180

# Helper functions
def calc_sum(landmarkList):
    tsum = 0
    for i in range(11, 33):
        tsum += (landmarkList[i].x * 480)  # Assuming you're using a fixed frame height of 480
    return tsum

def calc_dist(landmarkList):
    return (landmarkList[28].y * 640 - landmarkList[24].y * 640)  # Assuming a fixed width of 640

def isVisible(landmarkList):
    if (landmarkList[28].visibility > 0.7) and (landmarkList[24].visibility > 0.7):
        return True
    return False

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
drawing = mp.solutions.drawing_utils

# Load images for the green and red lights
im1 = cv2.imread('im1.jpg')
im2 = cv2.imread('im2.jpg')

currWindow = im1

# Capture video from webcam (uncomment if using webcam)
cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

while True:
    # Read from the webcam
    _, frm = cap.read()  # Capture frame from webcam
    if frm is None:
        break
    
    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
    res = pose.process(rgb)
    
    frm = cv2.blur(frm, (5, 5))  # Apply blur to reduce noise
    drawing.draw_landmarks(frm, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    if not inFramecheck:
        try:
            if isVisible(res.pose_landmarks.landmark):
                inFrame = 1
                inFramecheck = True
            else:
                inFrame = 0
        except:
            print("You are not visible at all")
    
    if inFrame == 1:
        if not isInit:
            playsound('greenlight.mp3')
            currWindow = im1
            startT = time.time()
            endT = startT
            dur = np.random.randint(3, 7)
            isInit = True

        if (endT - startT) <= dur:
            try:
                m = calc_dist(res.pose_landmarks.landmark)
                if m < thresh:
                    cPos += 1
                print("Current progress is: ", cPos)
            except:
                print("Not visible")
            
            endT = time.time()
        else:
            if cPos >= 100:
                print("WINNER")
                winner = 1
            else:
                if not isCinit:
                    isCinit = True
                    cStart = time.time()
                    cEnd = cStart
                    currWindow = im2
                    playsound('redLight.mp3')
                    userSum = calc_sum(res.pose_landmarks.landmark)

                if (cEnd - cStart) <= 5:
                    tempSum = calc_sum(res.pose_landmarks.landmark)
                    cEnd = time.time()
                    if abs(tempSum - userSum) > 150:
                        print("DEAD ", abs(tempSum - userSum))
                        isAlive = 0
                else:
                    isInit = False
                    isCinit = False
        # Draw progress circle
        cv2.circle(currWindow, ((55 + 6 * cPos), 280), 15, (0, 0, 255), -1)

        # Resize images before concatenating to ensure same size
        frm_resized = cv2.resize(frm, (800, 400))
        currWindow_resized = cv2.resize(currWindow, (800, 400))
        
        mainWin = np.concatenate((frm_resized, currWindow_resized), axis=0)
        cv2.imshow("Main Window", mainWin)
    
    else:
        cv2.putText(frm, "Please make sure you are fully in frame", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 4)
        cv2.imshow("Window", frm)

    if cv2.waitKey(1) == 27 or isAlive == 0 or winner == 1:
        cv2.destroyAllWindows()
        cap.release()
        break

    frm = cv2.blur(frm, (5, 5))

    if isAlive == 0:
        cv2.putText(frm, "You are Dead", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.imshow("Main Window", frm)

    if winner == 1:
        cv2.putText(frm, "You are Winner", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        cv2.imshow("Main Window", frm)

cv2.waitKey(0)
