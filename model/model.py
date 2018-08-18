"""YOLOv3 model implementation with Keras"""

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Conv2D, UpSampling2D, concatenate, BatchNormalization, ZeroPadding2D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Add
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from model.utils import squeeze
from model.dataset import get_classes, get_anchors, data_generator


def DarknetConv2D(*args, **kwargs):
    """darknet conv2D using l2 regularizer and the corresponding padding"""
    dark_conv_kwargs = {
        'kernel_regularizer': l2(5e-4),
        'padding': 'valid' if kwargs.get('strides') == 2 else 'same'
    }
    dark_conv_kwargs.update(kwargs)
    return Conv2D(*args, **dark_conv_kwargs)


def DarknetConv2D_BN_LeakyReLU(*args, **kwargs):
    """darknet conv2D using LeakyReLU and BatchNormalization"""
    some_kwargs = {'use_bias': False}
    some_kwargs.update(kwargs)
    return squeeze(
        DarknetConv2D(*args, **some_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1)
    )


def residual_block(x, num_filters, num_blocks):
    """The residual block

    x: tensor, the input of the residual block
    num_filters: int, the number of filters
    num_blocks: int, the number of the residual block
    Return: tensor, the output of the residual block
    """
    x = squeeze(
        ZeroPadding2D(((1, 0), (1, 0))),
        DarknetConv2D_BN_LeakyReLU(num_filters, (3, 3), strides=2)
    )(x)
    for i in range(num_blocks):
        residual_input = x
        x = squeeze(
            DarknetConv2D_BN_LeakyReLU(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_LeakyReLU(num_filters, (3, 3))
        )(residual_input)
        x = Add()([residual_input, x])

    return x


def darknet53(x):
    """The darknet53 feature extractor

    x: tensor, the input
    Returns: tuple of tensor, (feature1, feature2, feature3), three extracted features of different scales
    """
    x = DarknetConv2D_BN_LeakyReLU(32, (3, 3))(x)
    x = residual_block(x, 64, 1)
    x = residual_block(x, 128, 2)
    x = residual_block(x, 256, 8)
    feature3 = x
    x = residual_block(x, 512, 8)
    feature2 = x
    x = residual_block(x, 1024, 4)
    feature1 = x
    return feature1, feature2, feature3


def predict(x, num_filters, side_x=None, up_sampling=False):
    """Get the prediction from the extracted features

    x: tensor, extracted features
    num_filters: int, number of filters
    side_x: tensor, extracted features from previous scale output
    up_sampling: bool, boolean value indicating whether should up sampling side_x
    Return: tuple of tensor, (x, y), x is the extracted features from this scale, y is the output of this scale
    """
    if up_sampling:
        side_x = squeeze(
            DarknetConv2D_BN_LeakyReLU(num_filters, (1, 1)),
            UpSampling2D(size=(2, 2))
        )(side_x)
        x = concatenate([side_x, x])
    x = squeeze(
        DarknetConv2D_BN_LeakyReLU(num_filters, (1, 1)),
        DarknetConv2D_BN_LeakyReLU(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_LeakyReLU(num_filters, (1, 1)),
        DarknetConv2D_BN_LeakyReLU(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_LeakyReLU(num_filters, (1, 1))
    )(x)
    y = squeeze(
        DarknetConv2D_BN_LeakyReLU(num_filters * 2, (3, 3)),
        Conv2D(255, (1, 1), strides=1, padding='same')
    )(x)

    return x, y


def base_model(image_input):
    """the yolov3 base model

    image_input: tensor, shape=(H, W, 3)
    Returns: Model, the YOLOv3 model
    """
    feature1, feature2, feature3 = darknet53(image_input)

    x, y1 = predict(feature1, 512)
    x, y2 = predict(feature2, 256, x, up_sampling=True)
    x, y3 = predict(feature3, 128, x, up_sampling=True)

    return Model(image_input, [y1, y2, y3])


def train(annotation_path, classes_path, anchors_path, weights_path, log_dir):
    """Train YOLOv3 model

    annotation_path: string
    classes_path: string
    anchors_path: string
    weights_path: string
    log_dir: string
    """
    input_shape = (416, 416)
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    num_classes = len(class_names)
    model = training_model(input_shape, anchors,
                           num_classes=len(class_names),
                           weights_path=weights_path,
                           freeze_body=2)

    logging = TensorBoard(log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss',
                                 save_weights_only=True,
                                 save_best_only=True,
                                 period=3)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping('val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1
    with open(annotation_path) as f:
        annotations = f.readlines()
    np.random.seed(4)
    np.random.shuffle(annotations)
    num_val = int(len(annotations) * val_split)
    num_train = len(annotations) - num_val
    batch_size = 32

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
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

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
        model.save_weights(log_dir + 'trained_weights_stage_final.h5')


def preprocess_pred(pred, input_shape, anchors, num_classes):
    """preprocess yolo model prediction

    process x, y to be the proportion of x, y by W, H
    process w, h to be the proportion of w, h by W, H
    process box confidence to be in (0, 1)
    process the probability of all classes to be in (0, 1)

    pred: tensor, shape=(b, h, w, num_anchors * (5 + num_classes), prediction of yolo model
    input_shape: tensor, shape=(2,), wh
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    Return: list of tensor, (box_xy, box_wh, box_confidence, box_classes, raw_box_xy, raw_box_wh, grid)
        box_xy and box_wh shape=(b, h, w, num_anchors, 2), xywh is relative to input_shape
        box_confidence shape=(b, h, w, num_anchors, 1)
        box_class_probs shape=(b, h, w, num_anchors, num_classes)
        raw_box_xy and raw_box_wh shape=(b, h, w, num_anchors, 2), x is tx, y is ty, w is tw, h is th
        grid shape=(h, w, 1, 2)
    """
    float_type = K.dtype(pred)
    pred_shape = K.shape(pred)
    pred = K.reshape(pred, shape=(pred_shape[0], pred_shape[1], pred_shape[2], len(anchors), 5 + num_classes))
    anchor_tensor = K.constant(anchors, dtype=float_type)
    anchor_tensor = K.reshape(anchor_tensor, shape=(1, 1, 1, -1, 2))

    input_shape = K.cast(input_shape, dtype=float_type)

    h = K.shape(pred)[1]
    w = K.shape(pred)[2]
    grid_y = K.reshape(K.arange(0, h), shape=(h, 1, 1, 1))
    grid_y = K.tile(grid_y, [1, w, 1, 1])
    grid_x = K.reshape(K.arange(0, w), shape=[1, w, 1, 1])
    grid_x = K.tile(grid_x, [h, 1, 1, 1])
    grid = K.concatenate([grid_y, grid_x], axis=-1)
    grid = K.cast(grid, dtype=float_type)

    grid_shape = K.cast(K.shape(pred)[1:3], dtype=float_type)

    raw_box_xy = K.sigmoid(pred[..., 0:2])
    raw_box_wh = pred[..., 2:4]
    box_confidence = K.sigmoid(pred[..., 4:5])
    box_class_probs = K.sigmoid(pred[..., 5:])
    box_xy = (raw_box_xy + grid[..., ::-1]) / grid_shape[::-1]
    box_wh = K.exp(raw_box_wh) * anchor_tensor / input_shape

    return box_xy, box_wh, box_confidence, box_class_probs, raw_box_xy, raw_box_wh, grid


def box_iou(pred, y_true):
    """calculate the iou between every prediction box and true box

    pred: tensor, shape=(h, w, num_anchors, 4), all prediction boxes
    y_true: tensor, shape=(j, 4), list of true boxes from one image
    Return: tensor, shape=(h, w, num_anchors, j)
    """
    pred = K.expand_dims(pred, axis=-2)
    pred_xy = pred[..., :2]
    pred_wh = pred[..., 2:4]
    pred_xy_min = pred_xy - pred_wh / 2.0
    pred_xy_max = pred_xy + pred_wh / 2.0
    pred_area = pred_wh[..., 0] * pred_wh[..., 1]

    y_true = K.cast(y_true, dtype=K.dtype(pred))
    y_true = K.expand_dims(y_true, axis=0)
    y_true_xy = y_true[..., :2]
    y_true_wh = y_true[..., 2:4]
    y_true_xy_min = y_true_xy - y_true_wh / 2.0
    y_true_xy_max = y_true_xy + y_true_wh / 2.0
    y_true_area = y_true_wh[..., 0] * y_true_wh[..., 1]

    intersect_min = K.maximum(pred_xy_min, y_true_xy_min)
    intersect_max = K.minimum(pred_xy_max, y_true_xy_max)
    intersect_wh = intersect_max - intersect_min
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    iou = intersect_area / (pred_area + y_true_area - intersect_area)

    return iou


def loss(inputs, anchors, ignore_thresh=0.5, print_loss=False):
    """ Compute yolo loss
    inputs: list of tensor, [y1, y2, y3, y_true1, y_true2, y_true3], shape=(b, h, w, num_anchors, 5 + num_classes)
    anchors: array, shape=(N, 2), each anchor value is wh
    ignore_thresh: float, the ignore thresh
    print_loss: bool, whether should print loss

    Return: tensor, shape=(1,), the loss tensor
    """
    assert len(inputs) == 6, 'inputs should has six entry'
    predicts = inputs[:3]  # list of tensor
    labels = inputs[3:]  # list of tensor

    float_type = K.dtype(predicts[0])

    m = K.shape(predicts[0])[0]
    mf = K.cast(m, dtype=float_type)

    num_scales = len(predicts)
    input_shape = K.cast(K.shape(predicts[0])[1:3] * 32, dtype=float_type)[..., ::-1]  # wh
    anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    anchors = np.array(anchors, dtype=float_type)

    losses = 0

    for s in range(num_scales):
        y_true = K.cast(labels[s], dtype=float_type)
        true_mask = y_true[..., 4:5]
        true_mask_bool = K.cast(true_mask, dtype='bool')
        box_xy, box_wh, box_confidence, box_classes, \
        raw_box_xy, raw_box_wh, grid = preprocess_pred(predicts[s], input_shape, anchors[anchor_masks[s]])

        loss_scale = 2 - y_true[..., 2:3] * y_true[..., 3:4]

        raw_true_xy = y_true[..., :2] - grid[..., ::-1]
        raw_true_wh = K.log(y_true[..., 2:4] * input_shape / anchors[anchor_masks[s]])
        raw_true_wh = K.switch(true_mask, raw_true_wh, K.zeros_like(raw_true_wh, dtype=float_type))

        ignore_mask = tf.TensorArray(dtype=float_type, size=1, dynamic_size=True)
        box_xywh = K.concatenate([box_xy, box_wh], axis=-1)
        true_xywh = K.concatenate([raw_true_xy, raw_true_wh])

        def loop_body(b, ignore_mask):
            true_boxes = tf.boolean_mask(true_xywh[b, ...], mask=true_mask_bool[b, ..., 0])  # shape=[j, 4]
            iou = box_iou(box_xywh[b, ...], true_boxes)
            best_iou = K.max(iou, axis=-1, keepdims=True)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, dtype=float_type))

            return b + 1, ignore_mask

        _, ignore_mask = K.control_flow_ops.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()

        xy_loss = true_mask * loss_scale * K.binary_crossentropy(raw_box_xy, raw_true_xy, from_logits=True)
        wh_loss = true_mask * loss_scale * 0.5 * K.square(raw_box_wh - raw_true_wh)
        class_loss = true_mask * K.binary_crossentropy(box_classes, y_true[..., 5:], from_logits=True)
        confidence_loss = true_mask * K.binary_crossentropy(box_confidence, y_true[..., 4:5], from_logits=True) + \
                          (1 - true_mask) * K.binary_crossentropy(box_confidence, y_true[..., 4:5], from_logits=True) * ignore_mask

        losses += K.sum(xy_loss) / mf + \
                K.sum(wh_loss) / mf + \
                K.sum(class_loss) / mf + \
                K.sum(confidence_loss) / mf

        if print_loss:
            losses = tf.Print(losses, [losses, xy_loss, wh_loss, class_loss, confidence_loss, K.sum(ignore_mask)], message='loss: ')

    return losses


def training_model(input_shape, anchors, num_classes, weights_path, freeze_body=2, load_pretrained=False):
    """Create model for training

    input_shape: tuple of int, (W, H)
    anchors: array, shape=(N, 2)
    num_classes: integer
    weights_path: string
    freeze_body: int, 1 or 2, when 1, freezing the darknet part, when 2, freezing all layers except the 3 output layer
    load_pretrained: boolean

    return: The Model for training
    """
    image_input = Input(shape=[None, None, 3])
    base_model = base_model(image_input)
    w, h = input_shape
    scales = {0: 32, 1: 16, 2: 8}
    scales_count = len(anchors) // 3
    y_trues = [Input(shape=(h // scales[s], w // scales[s], scales_count, 5 + num_classes))
               for s in range(0, scales_count)]

    if load_pretrained:
        base_model.load_weights(weights_path, by_name=True, skip_mismatch=True)
        if freeze_body in [1, 2]:
            num = [185, len(base_model.layers) - 3][freeze_body - 1]
            for i in range(0, num):
                base_model.layers[i].trainable = False

    loss_layer = Lambda(loss,
                        name='yolo_loss',
                        output_shape=(1,),
                        arguments={
                            'anchors': anchors,
                            'ignore_thresh': 0.5,
                            'print_loss': True
                        })([*base_model.output, *y_trues])

    return Model([base_model.input, *y_trues], loss_layer)


def get_detected_boxes(predicts, image_shape, anchors, num_classes,
                       score_threshold=0.6, max_boxes=20, iou_threshold=0.5):
    """Filter ineffective predicts to get detected result

    predicts: list of tensor, each has shape=(1, h, w, num_anchors, 5 + num_classes)
    image_shape: tensor, shape=(2,), wh
    anchors: array, shape=(N, 2)
    num_classes: integer
    score_threshold: float
    max_boxes: integer
    iou_threshold: float
    Return: tuple of tensor, (boxes, scores, classes), each shape (N,4), (N,), (N,)
    """
    num_scales = len(predicts)
    input_shape = K.shape(predicts[0])[1:3] * 32
    input_shape = input_shape[::-1]
    boxes = []
    scores = []
    classes = []

    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    for i in range(num_scales):
        box_xy, box_wh, box_confidence, box_class_probs, _, _, _ = preprocess_pred(predicts[i], input_shape,
                                                                                   anchors[anchor_mask[i]], num_classes)

        # (1, h, w, num_anchors, 4), x_min, y_min, x_max, y_max, relative to image shape
        rescaled_boxes = rescale_pred_box(box_xy, box_wh, input_shape, image_shape)
        nms_boxes = K.concatenate([rescaled_boxes[..., 1:2], rescaled_boxes[..., 0:1],
                                   rescaled_boxes[..., 3:4], rescaled_boxes[..., 2:3]], axis=-1)
        nms_boxes = K.reshape(nms_boxes, shape=(-1, 4))
        rescaled_boxes = K.reshape(rescaled_boxes, shape=(-1, 4))
        box_scores = K.reshape(box_confidence * box_class_probs, shape=(-1, num_classes))

        max_boxes_tensor = K.constant(max_boxes, dtype='int32')
        mask = box_scores > score_threshold

        for c in range(num_classes):
            effective_boxes = tf.boolean_mask(nms_boxes, mask[..., c])
            masked_rescaled_boxes = tf.boolean_mask(rescaled_boxes, mask[..., c])
            effective_scores = tf.boolean_mask(box_scores[..., c], mask[..., c])
            nms_index = tf.image.non_max_suppression(effective_boxes, effective_scores,
                                                     max_boxes_tensor, iou_threshold=iou_threshold)
            effective_boxes = K.gather(masked_rescaled_boxes, nms_index)
            effective_scores = K.gather(effective_scores, nms_index)
            effective_classes = K.zeros_like(effective_scores, dtype='int32') + c
            boxes.append(effective_boxes)
            scores.append(effective_scores)
            classes.append(effective_classes)

    boxes = K.concatenate(boxes, axis=0)
    scores = K.concatenate(scores, axis=0)
    classes = K.concatenate(classes, axis=0)

    return boxes, scores, classes


def rescale_pred_box(pred_xy, pred_wh, input_shape, image_shape):
    """Rescale prediction box to image shape

    pred_xy: tensor, shape=(1, h, w, num_anchors, 2)
    pred_wh: tensor, shape=(1, h, w, num_anchors, 2)
    input_shape: tensor, shape=(2,), wh
    image_shape: tensor, shape=(2,), wh
    return: rescaled boxes, shape=(1, h, w, num_anchors, 4), x_min, y_min, x_max, y_max
    """
    input_shape = K.cast(input_shape, dtype=K.dtype(pred_xy))
    image_shape = K.cast(image_shape, dtype=K.dtype(pred_xy))
    new_shape = K.round(K.min(input_shape / image_shape) * image_shape)
    dxy = (input_shape - new_shape) / 2.0
    pred_xy = pred_xy * input_shape - dxy
    pred_wh = pred_wh * input_shape
    pred_xy_min = pred_xy - pred_wh / 2.0
    pred_xy_max = pred_xy + pred_wh / 2.0
    pred_xy_min = pred_xy_min / new_shape * image_shape
    pred_xy_max = pred_xy_max / new_shape * image_shape

    return K.concatenate([pred_xy_min, pred_xy_max], axis=-1)