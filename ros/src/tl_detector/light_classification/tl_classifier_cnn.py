from styx_msgs.msg import TrafficLight
import tensorflow as tf
import cv2
import numpy as np


class CNNTLStateDetector(object):
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = './frozen_inference_graph.pb'

    def __init__(self):

        # Define if printing debugging information
        try:
            self.DEBUGGING = DEBUGGING
        except:
            self.DEBUGGING = False

        """Initialization routines"""

        # Checking TensorFlow Version
        print('TensorFlow Version: %s' % str(tf.__version__))

        tf_config = tf.ConfigProto(log_device_placement=False) # If TRUE it will provide verbose reporting
        tf_config.operation_timeout_in_ms = 10000  # terminate if not returning in 10 seconds

        # Checking for a GPU
        if not tf.test.gpu_device_name():
            print('No GPU found. Please use a GPU to assure promply response from tl_detector.')
        else:
            print('Default GPU Device: %s'% tf.test.gpu_device_name())
            tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5 # We divide GPU capacity into two

        # Restoring ssd_mobilenet_v1_coco
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                self.detect_session = tf.Session(config=tf_config)

        # Restoring cnn for traffic light color detection
        self.state_graph = tf.Graph()
        with self.state_graph.as_default() as gr:
            self.state_session = tf.Session(config=tf_config)
            loader = tf.train.import_meta_graph('state_detector.meta')
            loader.restore(self.state_session, "./state_detector")
            self.out = gr.get_tensor_by_name("output_class:0")
            self.input_layer = gr.get_tensor_by_name("input_layer:0")
            self.keep_prob = gr.get_tensor_by_name("keep_prob:0")

    def load_image_into_numpy_array(self, image):
        try:
            (im_width, im_height) = image.size
            return np.array(image.getdata()).reshape(
                (im_height, im_width, 3)).astype(np.uint8)
        except:
            # If the previous procedure fails, we expect the
            # image is already a Numpy ndarray
            return image

    def get_classification(self, image):
        """
        Determines the color of the traffic light in a
        cropped portion of the image

        Args:
            image (cv::Mat): image containing the cropped
            image of a traffic light

        Returns:
            int: ID of traffic light color (as specified in styx_msgs/TrafficLight)

        """

        # Definite input and output Tensors for detection_graph
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        # Assuring the input is in RGB format
        RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Assuring the image is a Numpy array
        image_np = self.load_image_into_numpy_array(RGB_image)
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Detecting boxes, classes and scores in the image
        (boxes, scores, classes, num) = self.detect_session.run(
            [detection_boxes, detection_scores, detection_classes,
             num_detections],
            feed_dict={image_tensor: image_np_expanded})


        detections = list()
        for i in range(0, len(boxes[0])):
            # Scores are sorted, ignore matches with very low score
            # cClass 10 is a traffic light
            if scores[0][i] > 0.1 and classes[0][i] == 10:
                start_x = int(boxes[0][i][0] * image_np.shape[0])
                end_x = int(boxes[0][i][2] * image_np.shape[0])
                start_y = int(boxes[0][i][1] * image_np.shape[1])
                end_y = int(boxes[0][i][3] * image_np.shape[1])

                cut_image = image_np[start_x:end_x, start_y:end_y]
                detections.append([scores[0][i], cut_image])

        # We consider all the detections starting from the most confident ones
        # As soon as we have a confirmed traffic light color, we return it

        for score, cut_image in sorted(detections, reverse=True):
            if self.DEBUGGING:
                cv2.imshow('image', cut_image)
                cv2.waitKey(1)

            result = self.state_session.run(
                [self.out], feed_dict={self.input_layer: [cut_image],
                                       self.keep_prob: 1.0})
            if int(result[0][0]) == 0:
                return TrafficLight.RED
            elif int(result[0][0]) == 1:
                return TrafficLight.YELLOW
            elif int(result[0][0]) == 2:
                return TrafficLight.GREEN
            else:
                # If the result is unknown (class==3),
                # we pass to the following detection, if any left
                continue

        # We return unknown only if all detections failed
        return TrafficLight.UNKNOWN

class CNNTLClassifier(object):
    def __init__(self):
        """Initialization routine"""
        self.session = tf.Session()
        loader = tf.train.import_meta_graph('classifier.meta')
        loader.restore(self.session, "./classifier")
        graph = tf.get_default_graph()
        self.out = graph.get_tensor_by_name("output_class:0")
        self.input_layer = graph.get_tensor_by_name("input_layer:0")
        self.keep_prob = graph.get_tensor_by_name("keep_prob:0")
        self.counter = 0

    def get_classification(self, image):
        """
        Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Image is resized and transformed in HSV colorspace
        resized_image = cv2.resize(image, (80, 60))
        fixed_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

        # One shot detection on all the image
        result = self.session.run([self.out],
                                    feed_dict={self.input_layer: [fixed_image],
                                               self.keep_prob: 1.0})

        # We simply return a stop or go feedback
        if int(result[0][0]) == 1:
            return TrafficLight.RED
        else:
            return TrafficLight.GREEN
