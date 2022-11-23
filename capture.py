import cv2
import pandas as pd
from datetime import datetime
import os

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
users = pd.read_csv("users.csv")
id = int(input("ID: "))
if users[users['id'] == id].size == 0:
    name = input("Name: ")
    users = users.append({'id': id, 'name': name}, ignore_index=True)
    users.to_csv("users.csv", index=False, header=['id', 'name'])
    print(f'Register user {name} success!')
else:
    name = users[users['id'] == id].iloc[0]['name']
    print(f'Wellcom {name}!')

camera = cv2.VideoCapture(0)
count = 0

if not os.path.exists('faces'):
    os.makedirs('faces')

while cv2.waitKey(1) != ord('q'):
    connected, img = camera.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected_faces = face_detector.detectMultiScale(img_gray, scaleFactor=1.1, minSize=(50,50))

    for x, y, w, h in detected_faces:
        
        # Save face to folders
        face_img = img_gray[y:y + h, x: x+ w]
        if cv2.waitKey(1) == ord('s'):
            img_name = f'face.{id}.{datetime.now().microsecond}.jpg'
            cv2.imwrite(f'faces/{img_name}', face_img)
            count+= 1
            print(f'Take {count} photo!')

        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)

    cv2.imshow("Capture camera", img)

camera.release()
cv2.destroyAllWindows()
