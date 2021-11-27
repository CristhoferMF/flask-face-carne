from FrontalFaceDetector import FrontalFaceDetector

frontal = FrontalFaceDetector()
frontal.setImagePath("./storage/foto-small.jpg")
frontal.drawAll()
data = frontal.getData()
print(data)
frontal.showImage()
