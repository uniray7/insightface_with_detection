import cv2
import tkinter as tki
import sys
import time
import uuid, redis, json
from capture import Capturer
from recog_utils import get_feature

STATUS = {}
STATUS['face_is_valid'] = False
STATUS['valid_face'] = None

def showFrame():
  STATUS['face_is_valid'], imgtk, STATUS['valid_face'] = capturer.capAndAnal()
  lmain.imgtk = imgtk
  lmain.configure(image=imgtk)
  lmain.after(30, showFrame)

def takeSnapshot():
  name = inputTxt.get()
  if STATUS['face_is_valid']:
    feature = get_feature(STATUS['valid_face'])
    key = uuid.uuid1()
    value = json.dumps({'name': name, 'feature': feature.tolist()})

    redis_cli.set(key, value)
    print('success')
  print(name)


FRAME = None
redis_cli = redis.Redis(host='localhost', port=16380, decode_responses=True)
capturer = Capturer()

tkWindow = tki.Tk()
tkWindow.wm_title("Capture")
imgFrame = tki.Frame(tkWindow, width=600, height=500)
imgFrame.grid(row=0, column=0, padx=10, pady=2)

lmain = tki.Label(imgFrame)
lmain.grid(row=0, column=0)

controlFrame = tki.Frame(tkWindow, width=100, height=500)
controlFrame.grid(row=0, column=1, padx=10, pady=2)

inputTxt = tki.Entry(controlFrame)
inputTxt.grid(row=0, column=0, padx=10, pady=2)

btn = tki.Button(controlFrame, text="Capture Face!",command=takeSnapshot)
btn.grid(row=2, column=0, padx=10, pady=1)


showFrame()
tkWindow.mainloop() 
