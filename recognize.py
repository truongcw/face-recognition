import cv2
import pandas as pd

users = pd.read_csv('users.csv')
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create(threshold=500)
face_recognizer.read('lbphClassifier.yml')
camera = cv2.VideoCapture(0)



while cv2.waitKey(1) != ord('q'):
    connected, img = camera.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected_faces = face_detector.detectMultiScale(img_gray, scaleFactor=1.1, minSize=(50,50))

    for x, y, w, h in detected_faces:
        img_face = cv2.resize(img_gray[y:y + h, x:x + w], (250, 250))
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
        id, trust = face_recognizer.predict(img_face)
        if id != -1:
            print(trust)
            name = users[users['id'] == id].iloc[0]['name']
            cv2.putText(img, name, (x,y + h + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255))

    cv2.imshow("Recognize", img)

camera.release()
cv2.destroyAllWindows()