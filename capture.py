import cv2
import sys
from PIL import Image, ImageTk
import numpy as np
sys.path.append("..")
from face_detect import face_detector

HEAD_WIDTH = 270
HEAD_HEIGHT = 320

class Capturer:
  def __init__(self, source=0):
    self.capturer = cv2.VideoCapture(0)
    if not self.capturer.isOpened():
      print("cam not open")
      sys.exit(0)

    self.width = self.capturer.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    self.height = self.capturer.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    if self.width < 600 or self.height < 400:
      sys.exit(0)
  
    self.HEAD_x = int((self.width - HEAD_WIDTH)/2)
    self.HEAD_y = int((self.height -HEAD_HEIGHT)/2)
    self.HEAD_x_end = self.HEAD_x+HEAD_WIDTH
    self.HEAD_y_end = self.HEAD_y+HEAD_HEIGHT
    self.faceDetector = face_detector(thresh=0.998)

  def capAndAnal(self):
    validFace = None
    faceIsValid = False
    success, img = self.capturer.read()
    if not success:
      print(success)
      sys.exit(-1)

    cv2.rectangle(img, (self.HEAD_x, self.HEAD_y), (self.HEAD_x_end, self.HEAD_y_end), (255, 255, 255), 2)
    img_ext = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces, score = self.faceDetector.detect(np.expand_dims(img_ext, axis=0))
    for face_ctr, face in enumerate(faces):
      det = [0] * 4
      for i in range(4):
        det[i] = int(face[i])
      det[0] = det[0]-15
      det[1] = det[1]-15
      det[2] = det[2]+15
      det[3] = det[3]+15
      cv2.rectangle(img, (det[0], det[1]), (det[2], det[3]), (255, 255, 255), 2)
      if (det[2] - det[0] < 180) or (det[3] - det[1] < 220):
        continue
      if det[0] > self.HEAD_x and det[1] > self.HEAD_y and det[2] < self.HEAD_x_end and det[3] < self.HEAD_y_end:
        cv2.rectangle(img, (self.HEAD_x, self.HEAD_y), (self.HEAD_x_end, self.HEAD_y_end), (0, 0, 255), 2)
        faceIsValid = True
        validFace = img[det[1]:det[3], det[0]:det[2]]
        break
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    return faceIsValid, img, validFace
