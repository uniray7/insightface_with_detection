import numpy as np
import sys
import tensorflow as tf
                
sys.path.append("..")

from detect_utils import label_map_util
from detect_utils import visualization_utils_color as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './detect_models/frozen_inference_graph_face.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './detect_protos/face_label_map.pbtxt'

NUM_CLASSES = 2

class face_detector:
    def __init__(self, thresh = 0.8):
        self.thresh = thresh
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        self.graph = detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with detection_graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            self.sess = tf.Session(graph=detection_graph, config=config)
            self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            self.boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            self.scores = detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    def __enter__(self):
        self.sess.__enter__()

    def __exit__(self):
        self.sess.__exit__()

    def detect(self, frames):
        (boxes, scores, classes, num_detections) = self.sess.run(
            [self.boxes, self.scores, self.classes, self.num_detections],
            feed_dict={self.image_tensor: frames})
        # boxes[frame_num][top100boxes][ymin, xmin, ymax, xmax]
        # scores[frame_num][top100scores]
        boxes = np.squeeze(boxes, axis=0)
        scores = np.squeeze(scores, axis =0)
        height, width = frames.shape[1], frames.shape[2]
        qlified_num = (np.where(scores>self.thresh))[0].size
        # qlified_boxes = [([box:1X4], score),  ([box:1X4], score), ...]

        # box is [ymin, xmin, ymax, xmax] in ratio of frame size
        # reorder it into [xmin, ymin, xmax, ymax]
        # convert the ratio to pixel unit
        qlified_boxes = np.multiply(boxes[:qlified_num], [height, width, height, width])
        ymin, xmin, ymax, xmax = qlified_boxes.T
        qlified_boxes = np.asarray([xmin, ymin, xmax, ymax]).T        
        qlified_scores = scores[:qlified_num]
        return qlified_boxes, qlified_scores 

