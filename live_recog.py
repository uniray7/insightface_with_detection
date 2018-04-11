from __future__ import print_function
import pickle
import cv2
import tensorflow as tf
import sys, os
import numpy as np
import redis
import json
from face_detect import face_detector

import face_embedding
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='face model test')

    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--model', default='./models/model-r50-am-lfw/model,0', help='path to load model.')
    parser.add_argument('--gpu', default=1, type=int, help='gpu id')
    parser.add_argument('--det', default=2, type=int, help='mtcnn option, 2 means using R+O, else using O')
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
    args = parser.parse_args()

    redis_cli = redis.Redis(host='localhost', port=16380, decode_responses=True)

    eval_keys = redis_cli.keys('*')
    if eval_keys:
        tmp_eval_values = redis_cli.mget(eval_keys)
        eval_values = []
        for eval_num in xrange(len(tmp_eval_values)):
            eval_values.append(json.loads(tmp_eval_values[eval_num]))
    faceDetector = face_detector(thresh=0.3)
    rtsp = 'rtsp://192.168.1.87/stream1'
    cam = cv2.VideoCapture(rtsp)
    if not cam.isOpened():
        print("cam not open")
        sys.exit(0)
    fps = cam.get(cv2.cv.CV_CAP_PROP_FPS) 
    width = cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)   # float
    height = cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT) # float

    width = int(width)
    height= int(height)
    model = face_embedding.FaceModel(args)

    cv2.namedWindow('Live', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Live', 1200,600)

    frame_num = 0

    while(True):
        frame_num = frame_num+1
        if frame_num%50 == 0:
            print('try to renew member list!')
            new_eval_keys = redis_cli.keys('*')
            
            if new_eval_keys != eval_keys:
                print('renew!!')
                eval_keys = new_eval_keys
                if eval_keys:
                    tmp_eval_values = redis_cli.mget(eval_keys)
                    eval_values = []
                    for eval_num in xrange(len(tmp_eval_values)):
                        eval_values.append(json.loads(tmp_eval_values[eval_num]))
        success, img = cam.read()
        if not success:
            print(success)
            break
        print('frame: ', frame_num, frame_num%5)
        if frame_num%5 != 0:
            cv2.imshow('Live',img)
            continue

        img_ext = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces, _ = faceDetector.detect(np.expand_dims(img_ext, axis=0)) 
        crop_faces = []
        crop_dets = []
        for face_ctr, face in enumerate(faces):
            det = [0] * 4
            for i in xrange(4):
                det[i] = int(face[i])
            det[0] = det[0]-15
            det[1] = det[1]-15
            det[2] = det[2]+15
            det[3] = det[3]+15

            
            face = img[det[1]:det[3], det[0]:det[2]]

            feature = model.get_feature(face)

            if feature is None:
                continue
            if ((det[2]-det[0])<40) or ((det[3]-det[1])<50):
                continue

            cv2.rectangle(img, (det[0], det[1]), (det[2], det[3]), (255, 255, 255), 2)

            for eval_num in xrange(len(eval_keys)):
                eval_feature = np.asarray(eval_values[eval_num]['feature'])
                dist = np.sum(np.square(eval_feature-feature))
                sim = np.dot(eval_feature, feature.T)
                if dist<1.2 and sim>0.2:
                    cv2.rectangle(img, (det[0], det[1]), (det[2], det[3]), (0, 0, 255), 2)
                    cv2.putText(img, str(eval_values[eval_num]['name']) ,(det[0], det[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            cv2.imshow('Live',img)
        cv2.waitKey(10)

cv2.destroyAllWindows()

