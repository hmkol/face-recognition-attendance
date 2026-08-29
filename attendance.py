import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'ImagesAttendance'
images = []
classNames = []
encodeListKnown = []

if not os.path.exists(path):
    os.makedirs(path)

myList = os.listdir(path)
print(f'Loading images from {path}:', myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is None:
        print(f"Warning: Could not read image '{cl}'. Skipping.")
        continue
    
    img_rgb = cv2.cvtColor(curImg, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(img_rgb)
    
    if len(encodings) > 0:
        encodeListKnown.append(encodings[0])
        classNames.append(os.path.splitext(cl)[0])
        print(f"Loaded encoding for: {os.path.splitext(cl)[0]}")
    else:
        print(f"Warning: No face found in '{cl}'. Skipping.")

print(f'Total known faces encoded: {len(encodeListKnown)}')

def markAttendance(name):
    csv_file = 'attendance.csv'
    if not os.path.exists(csv_file):
        with open(csv_file, 'w') as f:
            f.writelines('Name, Time\n')

    with open(csv_file, 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0].strip() for line in myDataList if line.strip()]
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
            print(f"Attendance marked for {name} at {dtString}")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")

while cap.isOpened():
    success, img = cap.read()
    if not success or img is None:
        print("Failed to grab frame from webcam. Retrying...")
        cv2.waitKey(100)
        continue

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurrFrame = face_recognition.face_locations(imgS)
    encodesCurrFrame = face_recognition.face_encodings(imgS, facesCurrFrame)

    if len(encodeListKnown) > 0:
        for encodeFace, faceLoc in zip(encodesCurrFrame, facesCurrFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            # Match threshold check (face distance lower is better, usually < 0.6)
            if matches[matchIndex] and faceDis[matchIndex] < 0.55:
                name = classNames[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
                markAttendance(name)
            else:
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, "Unknown", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam - Press Q to Exit', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()