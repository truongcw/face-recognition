import cv2
import os
import numpy as np

lbph = cv2.face.LBPHFaceRecognizer_create(threshold=500)

def get_image_names():
    paths = [os.path.join('faces', p) for p in os.listdir('faces')]
    faces = []
    ids = []

    for path in paths:
        face_img = cv2.imread(path)
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        _, id, _, _ = path.split('.')
        faces.append(face_gray)
        ids.append(int(id))
    
    return np.array(ids), faces

ids, faces = get_image_names()
print('Start training...!')
lbph.train(faces, ids)
print('Training done!')
lbph.write('lbphClassifier.yml')
print('Finished!')
