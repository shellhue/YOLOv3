import numpy as np
import colorsys
from timeit import default_timer as timer
from PIL import ImageFont, ImageDraw

import keras.backend as K
from keras.layers import Input
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from model.model import get_detected_boxes, base_model, training_model
from model.utils import resize_image
from model.data import get_classes, get_anchors, data_generator


class YOLOv3(object):
    def __init__(self,
                 initial_weights_path,
                 is_training,
                 annotations_path=None,
                 anchors_path=None,
                 classes_path=None,
                 log_dir=None,
                 score_threshold=0.7,
                 iou_threshold=0.5,
                 input_shape=(416, 416)):
        assert len(initial_weights_path) > 0, "Initial weights path can not be empty!"
        if is_training:
            assert len(annotations_path) > 0, "Annotations path can not be empty when train!"
            self._annotations_path = annotations_path
        self._anchors_path = anchors_path
        self._classes_path = classes_path
        self._log_dir = log_dir if log_dir.endswith('/') else log_dir + '/'
        self._initial_weights_path = initial_weights_path
        self._is_training = is_training

        self._score_threshold = score_threshold
        self._iou_threshold = iou_threshold
        self._input_shape = input_shape
        self._image_shape_placeholder = K.placeholder((2,), dtype='int32')
        self._image_input = Input((None, None, 3), dtype='float32')
        self._anchors = get_anchors(self._anchors_path)
        self._max_boxes = 100
        self._class_names = get_classes(self._classes_path)
        self._sess = K.get_session()

        if not self._is_training:
            self._model = base_model(self._image_input)
            # load model weights
            print('######### start loading weights ##########')
            self._model.load_weights(self._initial_weights_path)
            assert self._model.layers[-1].output_shape[-1] == \
                   len(self._anchors) / len(self._model.outputs) * (len(self._class_names) + 5), \
                'Mismatch between model and given anchor and class sizes'
            print('######### finish loading weights ##########')

            # get detected boxes
            self._boxes, self._scores, self._classes = get_detected_boxes(self._model.outputs,
                                                                          self._image_shape_placeholder,
                                                                          self._anchors,
                                                                          len(self._class_names),
                                                                          self._score_threshold,
                                                                          self._max_boxes,
                                                                          self._iou_threshold)

        # config color for each class box
        print('######### start config boxing colors ##########')
        hsv_tuples = [(x / len(self._class_names), 1.0, 1.0) for x in range(len(self._class_names))]
        self._colors = list(map(lambda hsv: colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2]), hsv_tuples))
        self._colors = list(map(lambda rgb: (int(rgb[0] * 255.), int(rgb[1] * 255.), int(rgb[2] * 255.)), self._colors))
        np.random.seed(23132)
        np.random.shuffle(self._colors)
        np.random.seed(None)
        print('######### finish config boxing colors ##########')

    def train(self, use_focal_loss=False):
        input_shape = (416, 416)
        class_names = get_classes(self._classes_path)
        anchors = get_anchors(self._anchors_path)
        num_classes = len(class_names)
        model = training_model(input_shape, anchors,
                               num_classes=len(class_names),
                               weights_path=self._initial_weights_path,
                               freeze_body_mode='DARKNET',
                               use_focal_loss=use_focal_loss)
        print("model.inputs=========")
        print(model.inputs)
        print(model.outputs)
        print("model.outputs=========")
        logging = TensorBoard(self._log_dir)
        checkpoint = ModelCheckpoint(self._log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                     monitor='val_loss',
                                     save_weights_only=True,
                                     save_best_only=True,
                                     period=1)
        reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=3, verbose=1)
        early_stopping = EarlyStopping('val_loss', min_delta=0, patience=10, verbose=1)

        val_split = 0.1
        with open(self._annotations_path) as f:
            annotations = f.readlines()
        np.random.seed(4)
        np.random.shuffle(annotations)
        num_val = int(len(annotations) * val_split)
        num_train = len(annotations) - num_val
        batch_size = 8

        def yolo_loss(label, pred):
            return pred
        # first just train the three output layer
        if True:
            model.compile(
                Adam(lr=1e-3),
                loss={
                    'yolo_loss': yolo_loss
                })
            model.fit_generator(
                data_generator(annotations[:num_train], input_shape, batch_size, anchors,
                               num_classes, max_boxes=self._max_boxes),
                steps_per_epoch=max(1, num_train // batch_size),
                epochs=50,
                validation_data=data_generator(annotations[num_train:], input_shape, batch_size, anchors, num_classes),
                validation_steps=max(1, num_val // batch_size),
                initial_epoch=0,
                callbacks=[logging, checkpoint]
            )
            model.save_weights(self._log_dir + 'trained_weights_stage_1.h5')

        # fine tuning all layers
        if False:
            for l in range(len(model.layers)):
                model.layers[l].trainable = True
            model.compile(
                Adam(lr=1e-4),
                loss={
                    'yolo_loss': lambda label, pred: pred
                })
            model.fit_generator(
                data_generator(annotations[:num_train], input_shape, batch_size, anchors,
                               num_classes, max_boxes=self._max_boxes),
                steps_per_epoch=max(1, num_train // batch_size),
                epochs=10,
                validation_data=data_generator(annotations[num_train:], input_shape, batch_size, anchors, num_classes),
                validation_steps=max(1, num_val // batch_size),
                initial_epoch=2,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping]
            )
            model.save_weights(self._log_dir + 'trained_weights_stage_final.h5')

    def detect_image(self, image):
        start = timer()
        resized_input_image = resize_image(image, self._input_shape)
        image_data = np.array(resized_input_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, axis=0)
        boxes, scores, classes = self._sess.run([self._boxes, self._scores, self._classes],
                                                feed_dict={
                                                   self._model.input: image_data,
                                                   self._image_shape_placeholder: [image.size[0], image.size[1]],
                                                   K.learning_phase(): 0
                                               })

        # draw boxes
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 400

        for i, c in reversed(list(enumerate(classes))):
            predicted_class = self._class_names[c]
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
                draw.rectangle([left + t, top + t, right - t, bottom - t], outline=self._colors[c])

            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self._colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)

            del draw
        end = timer()
        print('detecting takes: {:.2f}s'.format(end - start))
        return image

    def close_session(self):
        self._sess.close()


