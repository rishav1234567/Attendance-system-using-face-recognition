import face_recognition
import cv2
import os
import numpy as np
from datetime import datetime

# Path to the folder containing images
path = 'pics'
images = []
classNames = []
myList = os.listdir(path)
print("Images found in directory:", myList)

# Load images and corresponding class names
for c1 in myList:
    curImg = cv2.imread(f'{path}/{c1}')
    images.append(curImg)
    classNames.append(os.path.splitext(c1)[0])
print("Class Names:", classNames)

# Function to find encodings for the images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:  # Ensure at least one face is detected
            encodeList.append(encode[0])
        else:
            print(f"Face not detected in image {img}")
    return encodeList

# Function to mark attendance in a CSV file
def markAttendance(name):
    # Ensure the CSV file exists, and if not, create it with headers
    if not os.path.isfile('Attendance.csv'):
        with open('Attendance.csv', 'w') as f:
            f.write('Name,Time,Date\n')
    
    # Read the existing attendance records
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        
        # Add new entry if the name is not already in the file
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            dStr = now.strftime('%d/%m/%Y')
            f.writelines(f'{name},{dtString},{dStr}\n')

# Encode known images
print("Encoding images...")
encodeListKnown = findEncodings(images)
print("Encoding complete!")

# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from webcam.")
        break

    # Resize and convert to RGB for processing
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Detect faces and compute encodings in the current frame
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceloc in zip(encodesCurFrame, facesCurFrame):
        # Compare the current encoding with known encodings
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print("Face Distances:", faceDis)

        # Find the best match
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print("Match Found:", name)
            
            # Scale back face locations to original frame size
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            
            # Draw rectangle and label on the frame
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            
            # Mark attendance
            markAttendance(name)

    # Display the webcam feed
    cv2.imshow('Webcam', img)
    
    # Break the loop on pressing 'Enter' (ASCII code 13)
    if cv2.waitKey(10) == 13:
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()