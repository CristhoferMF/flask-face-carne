import cv2
from flask import Flask, make_response
import numpy as np

haar_cascade_face = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt2.xml')
harr_eye_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route("/img")
def img():
    img_raw = cv2.imread("./storage/foto-small.jpg")
    # make gray
    gray_picture = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

    img_raw = face_detection(img_raw, gray_picture)
    # Response
    retval, buffer = cv2.imencode('.jpg', img_raw)
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = 'image/jpg'
    return response

def face_detection(img_raw, gray_picture):
    faces_rects = haar_cascade_face.detectMultiScale(gray_picture, scaleFactor = 1.2, minNeighbors = 5);
    # Let us print the no. of faces found
    for (fx,fy,fw,fh) in faces_rects:
        radius = fh//2
        face_x = int(fx+0.5*fw)
        face_y = int(fy+0.5*fh)
        cv2.circle(img_raw, (face_x, face_y), radius, (0, 0, 255), 1)
        detect_eyes(img_raw, gray_picture)
    return img_raw

def detect_eyes(img, img_gray, classifier=""):
    coords = harr_eye_cascade.detectMultiScale(img_gray, 1.3, 5)# detect eyes
    height = np.size(img, 0) # get face frame height
    for (x, y, w, h) in coords:
        if y+h > height/2: # pass if the eye is at the bottom
            pass
        else:
            radius = h//2
            eye_x = int(x+0.5*w)
            face_y = int(y+0.5*h)
            cv2.circle(img, (eye_x, face_y), radius, (0, 255, 0), 1)
