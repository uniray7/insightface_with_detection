import argparse
import tensorflow as tf
import face_embedding

def get_feature(img):
  parser = argparse.ArgumentParser(description='Create user')
  parser.add_argument('--image-size', default='112,112', help='')
  parser.add_argument('--model', default='./models/model-r50-am-lfw/model,0', help='path to load model.')
  parser.add_argument('--gpu', default=0, type=int, help='gpu id')
  parser.add_argument('--det', default=2, type=int, help='mtcnn option, 2 means using R+O, else using O')
  parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
  parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
  
  args = parser.parse_args()
  model = face_embedding.FaceModel(args)
  
  feature = model.get_feature(img)
  return feature

