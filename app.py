from flask import Flask, jsonify
from FrontalFaceDetector import FrontalFaceDetector
import cv2
import base64

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route("/img")
def image():
    FILE_PATH = "./storage/foto-small.jpg"
    frontal = FrontalFaceDetector()
    frontal.setImagePath(FILE_PATH)
    data = frontal.getData()
    img = frontal.drawAll()
    retval, buffer = cv2.imencode('.jpg', img)
    base64_string = base64.b64encode(buffer).decode('utf-8')
    data['imgBase64'] = "data:image/jpg;base64,"+base64_string
    return jsonify(data)