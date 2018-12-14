import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2 as cv
from pathlib import Path
import time

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

# Object detection imports
from utils import label_map_util
from utils import visualization_utils as vis_util

START = 131
END = 231

# What model to download : Tensorflow detection model zoo
#MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
#MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
MODEL_NAME = 'ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03'
#MODEL_NAME = 'ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03'
#MODEL_NAME = 'ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
#MODEL_NAME = 'faster_rcnn_nas_coco_2018_01_28'
#MODEL_NAME = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
#MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
#MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
#MODEL_NAME = 'ssd_inception_v2_coco_2018_01_28'
#MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
#MODEL_NAME = 'faster_rcnn_resnet50_coco_2018_01_28'
#MODEL_NAME = 'faster_rcnn_resnet101_kitti_2018_01_28'
#MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28'
#MODEL_NAME = 'faster_rcnn_resnet101_fgvc_2018_07_19'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

# Download Model
MODEL_PATH = Path("./",MODEL_FILE)
if MODEL_PATH.is_file() == False: # If model was already downloaded, don't download again
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())

# Load a (frozen) Tensorflow model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

# Detection
# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'cropped'
TEST_IMAGE_PATHS = [ 
    os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}.jpg'.format(i)) for i in range(0, 198) ]
    #'image5.png']
    #os.path.join(PATH_TO_TEST_IMAGES_DIR, '6.jpg')]
    #os.path.join(PATH_TO_TEST_IMAGES_DIR, '12.jpg')]
    #'{}.jpg'.format(i) for i in range(0, 7)]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

def main() :
    path='C:/Purduedurrrk/PretrainedModel/object_detection/crop_result'
    cnt = 0
    #with open('./result/crop_test_all/result.txt','w') as f:
    for image_path in TEST_IMAGE_PATHS:
        if not os.path.exists(image_path) :
            print('not exist')
            continue
        image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)
        plt.figure(figsize=IMAGE_SIZE)
        rgbimg = cv.cvtColor(image_np, cv.COLOR_BGR2RGB)
    
        label = []
        accuracy = []

        # Top 3 scores
        print(image_path)
        #writeStr = image_path + "\n"
        #f.write(writeStr)
        for i in range(0, 3):
            label.append(category_index[output_dict.get('detection_classes')[i].astype(np.uint8)]['name'])
            accuracy.append(output_dict.get('detection_scores')[i])
            #writeStr = "Label {} : {} ({})\n".format(i, label[i], accuracy[i])
            print("Label {} : {} ({})".format(i, label[i], accuracy[i]))
            #f.write(writeStr)

        # If "Bird" label has the highest accuracy, stop 
            '''if label[0] == 'bird':
                print('Bird is detected with accuracy ', accuracy[0])
                break'''

        # If in top 3 scores, "Bird" label has the accuracy more than 0.5, stop
        '''for i in range(0,3) :
            if label[i] == 'bird' and accuracy[i] >= 0.5:
                print('Bird is detected with accuracy ', accuracy[i])
                break'''

        # Store result images to path
        #cv.imshow("Image {}".format(cnt), rgbimg)
        cv.imwrite(os.path.join(path, '{}.jpg'.format(cnt)), rgbimg)
        cnt = cnt + 1

        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    cv.destroyAllWindows()

def mog(cap) :
    path='C:/Purduedurrrk/PretrainedModel/object_detection/result'
    count = 1
    while(count<(END+1)):
        if count==START:
            start_time = time.time()

        ret, frame = cap.read()
        _, original = cap.read()
        _, test = cap.read()

        if ret == False:
            break

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        image_np = load_image_into_numpy_array(image)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=8)
        plt.figure(figsize=IMAGE_SIZE)
        rgbimg = cv.cvtColor(image_np, cv.COLOR_BGR2RGB)
    
        label = []
        accuracy = []

        # Top 3 scores
        for i in range(0, 3):
            label.append(category_index[output_dict.get('detection_classes')[i].astype(np.uint8)]['name'])
            accuracy.append(output_dict.get('detection_scores')[i])
            print("Label ", i, " : ", label[i], "(", accuracy[i], ")")

        # If "Bird" label has the highest accuracy, stop 
        '''if label[0] == 'bird':
        print('Bird is detected with accuracy ', accuracy[0])
        break'''

        # If in top 3 scores, "Bird" label has the accuracy more than 0.5, stop
        for i in range(0, len(label)) :
            if label[i] == 'bird' and accuracy[i] >= 0.5:
                print('Bird is detected with accuracy ', accuracy[i])
                break

        # Store result images to path
        #cv.imshow("frame", rgbimg)
        #cv.imwrite(os.path.join(path, '{}.jpg'.format(count)), rgbimg)
        
        if count > END:
            end_time = time.time()
            break

        print(count)
        count = count + 1

        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    print("--- %.6s seconds ---"%(end_time-start_time))
    cap.release()
    cv.destroyAllWindows()