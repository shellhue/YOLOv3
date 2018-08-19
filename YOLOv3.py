import numpy as np
import argparse
import colorsys
from timeit import default_timer as timer
from PIL import Image, ImageFont, ImageDraw

import keras.backend as K
from keras.layers import Input
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from model.model import get_detected_boxes, base_model, training_model
from model.utils import resize_image
from model.dataset import get_classes, get_anchors, data_generator


parser = argparse.ArgumentParser(description='Darknet To Keras Converter.')
parser.add_argument('--use_retrained_weights', help='Whether should use retrained model weights')
parser.add_argument('--annotations_path', help='Path to annotations.')
parser.add_argument('--use_focal_loss', help='Whether use focal loss.')


class YOLOv3(object):
    def __init__(self,
                 pretrained_weights_path='model_data/yolo.h5',
                 retrained_weights_path='model_data/log/trained_weights_stage_final.h5',
                 anchors_path="model_data/yolo_anchors.txt",
                 classes_path="model_data/coco_classes.txt",
                 annotations_path='/home/shellhue/data/train.txt',
                 log_dir='model_data/log/',
                 use_retrained_weights=False,
                 score_threshold=0.7,
                 iou_threshold=0.45,
                 input_shape=(416, 416)):
        self.pretrained_weights_path = pretrained_weights_path
        self.retrained_weights_path = retrained_weights_path
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.annotations_path = annotations_path
        self.log_dir = log_dir
        self.use_retrained_weights = use_retrained_weights

        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.input_shape = input_shape
        self.image_shape_placeholder = K.placeholder((2,), dtype='int32')
        self.image_input = Input((None, None, 3), dtype='float32')
        self.anchors = get_anchors(self.anchors_path)
        self.max_boxes = 20
        self.class_names = get_classes(self.classes_path)
        self.sess = K.get_session()

        self.model = base_model(self.image_input)
        # load model weights
        self.model.load_weights(self.retrained_weights_path if self.use_retrained_weights else self.pretrained_weights_path)
        assert self.model.layers[-1].output_shape[-1] == \
            len(self.anchors) / len(self.model.outputs) * (len(self.class_names) + 5), \
            'Mismatch between model and given anchor and class sizes'

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

    def train(self, use_focal_loss=False):
        input_shape = (416, 416)
        class_names = get_classes(self.classes_path)
        anchors = get_anchors(self.anchors_path)
        num_classes = len(class_names)
        model = training_model(input_shape, anchors,
                               num_classes=len(class_names),
                               weights_path=self.pretrained_weights_path,
                               freeze_body=2,
                               use_focal_loss=use_focal_loss)

        logging = TensorBoard(self.log_dir)
        checkpoint = ModelCheckpoint(self.log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                     monitor='val_loss',
                                     save_weights_only=True,
                                     save_best_only=True,
                                     period=3)
        reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=3, verbose=1)
        early_stopping = EarlyStopping('val_loss', min_delta=0, patience=10, verbose=1)

        val_split = 0.1
        with open(self.annotations_path) as f:
            annotations = f.readlines()
        np.random.seed(4)
        np.random.shuffle(annotations)
        num_val = int(len(annotations) // 1000 * val_split)
        num_train = len(annotations) // 1000 - num_val
        batch_size = 8

        # first just train the three output layer
        if True:
            model.compile(
                Adam(lr=1e-3),
                loss={
                    'yolo_loss': lambda label, pred: pred
                })
            model.fit_generator(
                data_generator(annotations[:num_train], input_shape, batch_size, anchors, num_classes),
                steps_per_epoch=max(1, num_train // batch_size),
                epochs=50,
                validation_data=data_generator(annotations[num_train:], input_shape, batch_size, anchors, num_classes),
                validation_steps=max(1, num_val // batch_size),
                initial_epoch=0,
                callbacks=[logging, checkpoint]
            )
            model.save_weights(self.log_dir + 'trained_weights_stage_1.h5')

        # fine tuning all layers
        if True:
            for l in range(len(model.layers)):
                model.layers[l].trainable = True
            model.compile(
                Adam(lr=1e-4),
                loss={
                    'yolo_loss': lambda label, pred: pred
                })
            model.fit_generator(
                data_generator(annotations[:num_train], input_shape, batch_size, anchors, num_classes),
                steps_per_epoch=max(1, num_train // batch_size),
                epochs=100,
                validation_data=data_generator(annotations[num_train:], input_shape, batch_size, anchors, num_classes),
                validation_steps=max(1, num_val // batch_size),
                initial_epoch=50,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping]
            )
            model.save_weights(self.log_dir + 'trained_weights_stage_final.h5')

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


def _main(args):
    yolo = YOLOv3(annotations_path=str(args.annotations_path), use_retrained_weights=bool(args.use_retrained_weights))
    yolo.train(use_focal_loss=bool(args.use_focal_loss))


if __name__ == '__main__':

    _main(parser.parse_args())
