import numpy as np
import colorsys
from timeit import default_timer as timer
from PIL import Image, ImageFont, ImageDraw

import keras.backend as K
from keras.layers import Input
from keras.models import load_model

from model.model import get_detected_boxes, base_model
from model.dataset import get_anchors, get_classes
from model.utils import resize_image

import tensorflow as tf


class YOLOv3(object):
    def __init__(self,
                 model_path="model_data/yolo.h5",
                 anchor_path="model_data/yolo_anchors.txt",
                 classes_path="model_data/coco_classes.txt",
                 score_threshold=0.7,
                 iou_threshold=0.45,
                 input_shape=(416, 416)):
        self.model_path = model_path
        self.anchor_path = anchor_path
        self.classes_path = classes_path

        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.input_shape = input_shape
        self.image_shape_placeholder = K.placeholder((2,), dtype='int32')
        self.image_input = Input((None, None, 3), dtype='float32')
        self.anchors = get_anchors(self.anchor_path)
        self.max_boxes = 20
        self.class_names = get_classes(self.classes_path)
        self.sess = K.get_session()

        # load model
        try:
            self.model = load_model(self.model_path, compile=False)
        except:
            self.model = base_model(self.image_input)
            self.model.load_weights(self.model_path)
        else:
            assert self.model.layers[-1].output_shape[-1] == \
                len(self.anchors) / len(self.model.outputs) * (len(self.class_names) + 5), \
                'Mismatch between model and given anchor and class sizes'
        weights = self.model.get_weights()
        # get detected boxes
        self.boxes, self.scores, self.classes = get_detected_boxes(self.model.outputs,
                                                                   self.image_shape_placeholder,
                                                                   self.anchors,
                                                                   len(self.class_names),
                                                                   self.score_threshold,
                                                                   self.max_boxes,
                                                                   self.iou_threshold)

        # config color for each class box
        hsv_tuples = [(x / len(self.class_names), 1.0, 1.0) for x in range(len(self.class_names))]
        self.colors = list(map(lambda hsv: colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2]), hsv_tuples))
        self.colors = list(map(lambda rgb: (int(rgb[0] * 255.), int(rgb[1] * 255.), int(rgb[2] * 255.)), self.colors))
        np.random.seed(23132)
        np.random.shuffle(self.colors)
        np.random.seed(None)

    def detect_image(self, image):
        start = timer()
        resized_input_image = resize_image(image, self.input_shape)
        image_data = np.array(resized_input_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, axis=0)
        print('input image shape:', image_data.shape)
        boxes, scores, classes = self.sess.run([self.boxes, self.scores, self.classes],
                                               feed_dict={
                                                   self.model.input: image_data,
                                                   self.image_shape_placeholder: [image.size[0], image.size[1]],
                                                   K.learning_phase(): 0
                                               })

        # draw boxes
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 400

        for i, c in reversed(list(enumerate(classes))):
            predicted_class = self.class_names[c]
            box = boxes[i]
            score = scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)

            label_size = draw.textsize(label, font)

            left, top, right, bottom = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], int(np.floor(right + 0.5)))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for t in range(thickness):
                draw.rectangle([left + t, top + t, right - t, bottom - t], outline=self.colors[c])

            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)

            del draw
        end = timer()
        print('detecting takes: %.9f', end - start)
        return image

    def close_session(self):
        self.sess.close()


tensor = tf.constant([100, 22])
tensorb = tensor * 2

tensor = tf.Print(tensor, [tensor, tensorb, tf.constant(100)], message='tensor print test: ')

with tf.Session() as sess:
    print(sess.run([tensor, tensor]))