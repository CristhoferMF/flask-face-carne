import cv2
import os
from PIL import Image
import numpy as np

class FrontalFaceDetector:
    haar_cascade_face = cv2.CascadeClassifier(
        'data/haarcascades/haarcascade_frontalface_alt2.xml')
    harr_eye_cascade = cv2.CascadeClassifier(
        'data/haarcascades/haarcascade_eye.xml')
    harr_mouth_cascade = cv2.CascadeClassifier(
        'data/haarcascades/haarcascade_mcs_mouth.xml')
    img_path = ''
    IMG_DATA = {}
    img = ''
    gray_img = ''

    def __init__(self, img_path=""):
        self.img_path = img_path

    def setImagePath(self, img_path):
        self.img_path = img_path
        self.IMG_DATA['size'] = os.path.getsize(self.img_path)
        self.createImage()
        self.IMG_DATA['height'], self.IMG_DATA['width'], channels = self.img.shape
        self.setImageDataResolucion()

    def createImage(self):
        self.img = cv2.imread(self.img_path)
        self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def setImageDataResolucion(self):
        p_img = Image.open(self.img_path)
        self.IMG_DATA['resolucion'] = p_img.info['dpi']

    def getData(self):
        return self.IMG_DATA

    def calculateFaceRects(self):
        faces_rects = self.haar_cascade_face.detectMultiScale(
            self.gray_img, scaleFactor=1.2, minNeighbors=5)
        return faces_rects[0]

    def calculateEyesPosition(self):
        eye_coords = self.harr_eye_cascade.detectMultiScale(
            self.gray_img, 1.3, 5)  # detect eyes
        height = self.IMG_DATA['height']  # get face frame height
        new_eye_coords = []
        for (x, y, w, h) in eye_coords:
            if y+h > height/2:  # pass if the eye is at the bottom
                pass
            else:
                new_eye_coords.append((x, y, w, h))
                self.calculateCenterEyesPosition((x, y, w, h))
        return new_eye_coords

    def calculateMouthPosition(self):
        half_height = int(self.IMG_DATA['height']//2.3)
        cut_img = self.gray_img[half_height:, :]
        mouth_rects = self.harr_mouth_cascade.detectMultiScale(cut_img, 1.3, 5)[
            0]
        [x, y, w, h] = mouth_rects
        y = int(y - 0.15*h) + half_height
        self.calculateCenterMouthPosition([x, y, w, h])
        return [x, y, w, h]

    def calculateCenterMouthPosition(self, mouth_coords):
        [x, y, w, h] = mouth_coords
        center_x = int(x+0.5*w)
        center_y = int(y+0.5*h)
        self.IMG_DATA['mouthX'] = center_x
        self.IMG_DATA['mouthY'] = center_y
        return center_x, center_y

    def calculateCenterEyesPosition(self, eye_coords):
        (x, y, w, h) = eye_coords
        center_x = int(x+0.5*w)
        center_y = int(y+0.5*h)
        if (x+w < self.IMG_DATA['width']/2):
            self.IMG_DATA['leftEyeX'] = center_x
            self.IMG_DATA['leftEyeY'] = center_y
        else:
            self.IMG_DATA['rightEyeX'] = center_x
            self.IMG_DATA['rightEyeY'] = center_y
        return center_x, center_y

    def calculateAll(self):
        self.calculateFaceRects()
        self.calculateEyesPosition()
        self.calculateMouthPosition()

    def drawEyes(self):
        eye_coords = self.calculateEyesPosition()
        for (x, y, w, h) in eye_coords:
            radius = h//2
            eye_x = int(x+0.5*w)
            eye_y = int(y+0.5*h)
            cv2.circle(self.img, (eye_x, eye_y), radius, (0, 255, 0), 1)

    def drawFace(self):
        faces_rects = self.calculateFaceRects()
        [fx, fy, fw, fh] = faces_rects
        radius = fh//2
        face_x = int(fx+0.5*fw)
        face_y = int(fy+0.5*fh)
        cv2.circle(self.img, (face_x, face_y), radius, (0, 0, 255), 1)

    def drawAll(self):
        self.drawFace()
        self.drawEyes()
        self.drawMouth()
    def drawMouth(self):
        mouth_coords = self.calculateMouthPosition()
        [x, y, w, h] = mouth_coords
        cv2.rectangle(self.img, (x, y), (x+w, y+h), (255, 0, 0), 1)

    def showImage(self, title="my-image"):
        # show image
        cv2.imshow(title, self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
