# import face_recognition
# import cv2
# import os
# import numpy as np
# from datetime import datetime
# import sys

# script_dir = os.path.dirname(os.path.abspath(__file__))
# img_dir = os.path.join(script_dir, 'pics')
# csv_path = os.path.join(script_dir, 'Attendance.csv')
# images = []
# classNames = []
# try:
#     myList = os.listdir(img_dir)
#     print("Images found in directory:", myList)
# except Exception as e:
#     print(f"Error reading image directory: {e}")
#     sys.exit(1)

# # Load images and corresponding class names
# for c1 in myList:
#     img_path = os.path.join(img_dir, c1)
#     curImg = cv2.imread(img_path)
#     if curImg is None:
#         print(f"Warning: Could not read image {img_path}")
#         continue
#     images.append(curImg)
#     classNames.append(os.path.splitext(c1)[0])
# print("Class Names:", classNames)

# # Function to find encodings for the images
# def findEncodings(images, classNames):
#     encodeList = []
#     validClassNames = []
#     for img, name in zip(images, classNames):
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encode = face_recognition.face_encodings(img)
#         if encode:
#             encodeList.append(encode[0])
#             validClassNames.append(name)
#         else:
#             print(f"Face not detected in image for {name}")
#     return encodeList, validClassNames

# # Function to mark attendance in a CSV file
# def markAttendance(name, csv_path):
#     # Ensure the CSV file exists, and if not, create it with headers
#     if not os.path.isfile(csv_path):
#         with open(csv_path, 'w') as f:
#             f.write('Name,Time,Date\n')

#     now = datetime.now()
#     dtString = now.strftime('%H:%M:%S')
#     dStr = now.strftime('%d/%m/%Y')

#     # Read the existing attendance records
#     with open(csv_path, 'r+') as f:
#         myDataList = f.readlines()
#         # Check for today's attendance (case-insensitive)
#         entries = [line.strip().split(',') for line in myDataList[1:] if line.strip()]
#         already_marked = any((entry[0].strip().upper() == name.upper() and entry[2].strip() == dStr) for entry in entries)
#         if not already_marked:
#             f.writelines(f'{name},{dtString},{dStr}\n')

# print("Encoding images...")
# encodeListKnown, classNames = findEncodings(images, classNames)
# print("Encoding complete!")

# # Start webcam capture
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     sys.exit(1)

# print("Press 'q' to exit webcam.")
# try:
#     while True:
#         success, img = cap.read()
#         if not success:
#             print("Failed to capture image from webcam.")
#             break

#         # Resize and convert to RGB for processing
#         imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
#         imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

#         # Detect faces and compute encodings in the current frame
#         facesCurFrame = face_recognition.face_locations(imgS)
#         encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

#         for encodeFace, faceloc in zip(encodesCurFrame, facesCurFrame):
#             # Compare the current encoding with known encodings
#             matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
#             faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
#             print("Face Distances:", faceDis)

#             # Find the best match
#             matchIndex = np.argmin(faceDis)
#             if matches[matchIndex]:
#                 name = classNames[matchIndex].upper()
#                 print("Match Found:", name)

#                 # Scale back face locations to original frame size
#                 y1, x2, y2, x1 = faceloc
#                 y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

#                 # Draw rectangle and label on the frame
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
#                 cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

#                 # Mark attendance (case-insensitive, daily)
#                 markAttendance(name, csv_path)

#         # Display the webcam feed
#         cv2.imshow('Webcam', img)

#         # Break the loop on pressing 'q' (case-insensitive)
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break
# except Exception as e:
#     print(f"Error during webcam loop: {e}")
# finally:
#     cap.release()
#     cv2.destroyAllWindows()





import face_recognition
import cv2
import os
import numpy as np
from datetime import datetime

# Folder containing known face images
path = "pics"

images = []
classNames = []
markedNames = set()

# Load images
myList = os.listdir(path)
print("Images Found:", myList)

for file in myList:
    img = cv2.imread(os.path.join(path, file))
    if img is not None:
        images.append(img)
        classNames.append(os.path.splitext(file)[0])

print("Known Persons:", classNames)


# Generate face encodings
def findEncodings(images):
    encodeList = []

    for img in images:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        encodings = face_recognition.face_encodings(rgb)

        if len(encodings) > 0:
            encodeList.append(encodings[0])
        else:
            print("No face found in one of the images.")

    return encodeList


# Mark attendance only once
def markAttendance(name):
    with open("Attendance.csv", "a+") as f:
        f.seek(0)
        data = f.readlines()

        nameList = []

        for line in data:
            entry = line.strip().split(",")
            if len(entry) > 0:
                nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            timeString = now.strftime("%H:%M:%S")
            dateString = now.strftime("%d/%m/%Y")

            f.write(f"\n{name},{timeString},{dateString}")

            print(f"{name} Attendance Marked")
        else:
            print(f"{name} Already Present in CSV")


# Encode all known faces
encodeListKnown = findEncodings(images)
print("Encoding Complete")

# Start webcam
cap = cv2.VideoCapture(0)

while True:

    success, img = cap.read()

    if not success:
        print("Failed to read from webcam.")
        break

    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(imgSmall)
    encodesCurrentFrame = face_recognition.face_encodings(
        imgSmall,
        facesCurrentFrame
    )

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):

        matches = face_recognition.compare_faces(
            encodeListKnown,
            encodeFace
        )

        faceDistance = face_recognition.face_distance(
            encodeListKnown,
            encodeFace
        )

        matchIndex = np.argmin(faceDistance)

        if matches[matchIndex]:

            name = classNames[matchIndex].upper()

            y1, x2, y2, x1 = faceLoc

            y1 *= 4
            x2 *= 4
            y2 *= 4
            x1 *= 4

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.rectangle(
                img,
                (x1, y2 - 35),
                (x2, y2),
                (0, 255, 0),
                cv2.FILLED
            )

            cv2.putText(
                img,
                name,
                (x1 + 6, y2 - 6),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 255, 255),
                2
            )

            # Prevent duplicate attendance during runtime
            if name not in markedNames:
                markAttendance(name)
                markedNames.add(name)

    cv2.imshow("Attendance System", img)

    # Press Enter or ESC to exit
    key = cv2.waitKey(1)

    if key == 13 or key == 27:
        break

cap.release()
cv2.destroyAllWindows()