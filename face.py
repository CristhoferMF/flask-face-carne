import cv2
import os
from PIL import Image
import numpy as np

haar_cascade_face = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt2.xml')
harr_eye_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')
harr_mouth_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_mcs_mouth.xml')

PATH_FILE = "./storage/foto-small.jpg"
IMG_DATA = {}
#file size
IMG_DATA['size'] = os.path.getsize(PATH_FILE)
#img data
img = cv2.imread("./storage/foto-small.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img height, width, channels
IMG_DATA['height'], IMG_DATA['width'], channels = img.shape 
#dpi
p_img = Image.open(PATH_FILE)
IMG_DATA['resolucion'] = p_img.info['dpi']

def get_faces_rects(gray_img):
    faces_rects = haar_cascade_face.detectMultiScale(gray_img, scaleFactor = 1.2, minNeighbors = 5)
    return faces_rects[0]

def draw_face(img, gray_img):
    faces_rects = get_faces_rects(gray_img)
    [fx,fy,fw,fh] = faces_rects
    radius = fh//2
    face_x = int(fx+0.5*fw)
    face_y = int(fy+0.5*fh)
    cv2.circle(img, (face_x, face_y), radius, (0, 0, 255), 1)

def get_eyes_position(coords):
    [x,y,w,h] = coords
    center_x = int(x+0.5*w)
    center_y = int(y+0.5*h)
    if (x+w < IMG_DATA['width']/2 ): 
        IMG_DATA['leftEyeX'] = center_x
        IMG_DATA['leftEyeY'] = center_y
    else:
        IMG_DATA['rightEyeX'] = center_x
        IMG_DATA['rightEyeY'] = center_y
    return center_x, center_y

def get_mouth_position (coords):
    [x,y,w,h] = coords
    center_x = int(x+0.5*w)
    center_y = int(y+0.5*h)
    IMG_DATA['mouthX'] = center_x
    IMG_DATA['mouthY'] = center_y
    return center_x, center_y

def draw_eyes(img, gray_img):
    eye_coords = harr_eye_cascade.detectMultiScale(gray_img, 1.3, 5)# detect eyes
    height = IMG_DATA['height'] # get face frame height
    for (x, y, w, h) in eye_coords:
        if y+h > height/2: # pass if the eye is at the bottom
            pass
        else:
            get_eyes_position([x,y,w,h])
            radius = h//2
            eye_x = int(x+0.5*w)
            eye_y = int(y+0.5*h)
            cv2.circle(img, (eye_x, eye_y), radius, (0, 255, 0), 1)
            
def get_mouth_rects(gray_img):
    half_height = int(IMG_DATA['height']//2.3)
    cut_img = gray_img[half_height:, :]
    mouth_rects = harr_mouth_cascade.detectMultiScale(cut_img, 1.3, 5)[0]
    [x,y,w,h] = mouth_rects
    y = int(y - 0.15*h) + half_height
    return [x,y,w,h]

def draw_mouth(img, gray_img):
    mouth_rects = get_mouth_rects(gray_img)
    get_mouth_position(mouth_rects)
    [x,y,w,h] = mouth_rects
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)

draw_face(img, gray_img)
draw_eyes(img, gray_img)
draw_mouth(img, gray_img)

print(IMG_DATA)
# show image
cv2.imshow("my-image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
