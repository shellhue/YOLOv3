"""Data prepare and preprocess"""

import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from PIL import Image
from model.utils import rand


def true_boxes_to_labels(true_boxes, input_shape, anchors, num_classes):
    """Preprocess true boxes from annotation to labels

    true_boxes: array, shape=(m, T, 5), not tensor
        absolute x_min, y_min, x_max, y_max, class_id relative to input shape
    input_shape: array like, shape=(H, W), H and W are multiples of 32, not tensor
    anchors: array, shape=(N, 2), 2 is WH, not tensor
    num_classes: integer, number of classes, not tensor
    Return: array, list of different scale y_true which has the same shape of YOLOv3
        output of the corresponding scale, each y_true has shape=[m, H, W, num_anchors, num_classes + 5]
    """
    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    anchors = np.array(anchors, dtype='float32').reshape(-1, 2)
    H, W = input_shape
    assert H % 32 == 0 and W % 32 == 0, 'input_shape must be multiples of 32'
    assert np.max(true_boxes[..., 4]) < num_classes, 'class id must be less than num_classes'
    m = true_boxes.shape[0]
    num_output = len(anchors) // 3
    scale = {0: 32, 1: 16, 2: 8}
    anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    y_trues = [np.zeros([m, H // scale[i], W // scale[i], 3, num_classes + 5], dtype='float32')
               for i in range(0, num_output)]
    box_xy_min = true_boxes[..., 0:2]
    box_xy_max = true_boxes[..., 2:4]
    box_classes = true_boxes[..., 4:5].astype('int32')
    box_xy = (box_xy_max + box_xy_min) / 2.0  # shape=(m, num_boxes, 2)
    box_wh = (box_xy_max - box_xy_min)  # shape=(m, num_boxes, 2)

    anchors = np.expand_dims(anchors, 0)  # shape=(1, num_anchors, 2)
    anchors_maxes = anchors / 2.0  # shape=(1, num_anchors, 2)
    anchors_mins = -anchors_maxes  # shape=(1, num_anchors, 2)
    anchor_area = anchors[..., 0] * anchors[..., 1]  # shape=(1, num_anchors)

    true_boxes_mask = box_wh[..., 0] > 0  # shape=(m, num_boxes)

    for b in range(0, m):
        true_boxes_wh = box_wh[b][true_boxes_mask[b]]  # shape=(num_true_boxes, 2)
        true_boxes_wh_o = true_boxes_wh
        true_boxes_xy = box_xy[b][true_boxes_mask[b]]  # shape=(num_true_boxes, 2)
        true_boxes_classes = box_classes[b][true_boxes_mask[b]]  # shape=(num_true_boxes, 1)

        if len(true_boxes_wh) == 0:
            continue
        true_boxes_wh = np.expand_dims(true_boxes_wh, -2)  # shape=(num_true_boxes, 1, 2)
        true_boxes_area = true_boxes_wh[..., 0] * true_boxes_wh[..., 1]  # shape=(num_true_boxes, 1)
        box_maxes = true_boxes_wh / 2.0  # shape=(num_true_boxes, 1, 2)
        box_mins = -box_maxes  # shape=(num_true_boxes, 1, 2)

        intersect_min = np.maximum(box_mins, anchors_mins)  # shape=(num_true_boxes, num_anchors, 2)
        intersect_max = np.minimum(box_maxes, anchors_maxes)  # shape=(num_true_boxes, num_anchors, 2)
        intersect_wh = intersect_max - intersect_min  # shape=(num_true_boxes, num_anchors, 2)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]  # shape=(num_true_boxes, num_anchors)

        iou = intersect_area / (anchor_area + true_boxes_area - intersect_area)  # shape=(num_true_boxes, N)

        best_iou = np.argmax(iou, axis=-1)  # shape=(num_true_boxes)

        for t, a in enumerate(best_iou):
            for l in range(0, num_output):
                if a in anchor_masks[l]:
                    cell_w, cell_h = true_boxes_xy[t].astype('int32') // scale[l]
                    x, y = true_boxes_xy[t] / input_shape[::-1]
                    w, h = true_boxes_wh_o[t] / input_shape[::-1]
                    c = true_boxes_classes[t][0]
                    i = anchor_masks[l].index(a)
                    y_trues[l][b][cell_h][cell_w][i][0:4] = [x, y, w, h]
                    y_trues[l][b][cell_h][cell_w][i][4] = 1
                    y_trues[l][b][cell_h][cell_w][i][5+c] = 1

    return y_trues


def get_classes(classes_path):
    """Read classes info from classes path

    classes_path: string, classes path
    Return: array, array of class names
    """
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [l.strip() for l in class_names]

    return class_names


def get_anchors(anchor_path):
    """Load all anchors form path

    anchor_path: string
    Return: array, shape=[N, 2]
    """
    with open(anchor_path) as f:
        anchors = f.readline()
    anchors = anchors.split(',')
    anchors = [a.strip() for a in anchors]
    anchors = np.array(anchors, dtype='float32').reshape([-1, 2])
    return anchors


def get_augmented_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    """Get real-time data augmentation

    annotation_line: string, one example data
    input_shape: tuple of integer, (H, W)
    random: boolean
    max_boxes: integer
    jitter: float
    hue: float
    sat: float
    val: float
    proc_img: boolean
    return: image_data, box_data, image data shape=[H, W, 3], box data shape=[max_boxes, 5]
    """
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

    if not random:
        # resize image
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        image_data = 0
        if proc_img:
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)/255.

        # correct boxes
        box_data = np.zeros((max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            if len(box) > max_boxes:
                box = box[:max_boxes]
            box[:, [0, 2]] = box[:, [0, 2]]*scale + dx
            box[:, [1, 3]] = box[:, [1, 3]]*scale + dy
            box_data[:len(box)] = box

        return image_data, box_data

    # resize image
    new_ar = w/h * rand(1 - jitter, 1 + jitter)/rand(1 - jitter, 1 + jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand() < .5
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)  # numpy array, 0 to 1

    # correct boxes
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]]*nw/iw + dx
        box[:, [1, 3]] = box[:, [1, 3]]*nh/ih + dy
        if flip:
            box[:, [0,2]] = w - box[:, [2,0]]
        box[:, 0: 2][box[:, 0: 2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)] # discard invalid box
        if len(box) > max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data


def data_generator(annotations, input_shape, batch_size, anchors, num_classes, max_boxes=20):
    """Sample generator, can infinitely generate sample data

    annotations: array, list of annotation string
    input_shape: tuple of int, (h, w)
    batch_size: int
    anchors: array, shape=[N, 2)
    num_classes: int
    max_boxes: int
    return: prepared input data
    """
    annotations = np.array(annotations)
    n = len(annotations)
    i = 0
    while True:
        image_datas = []
        boxes = []
        for b in range(0, batch_size):
            if i == 0:
                np.random.shuffle(annotations)
            annotation = annotations[i]
            image_data, box_data = get_augmented_data(annotation, input_shape, random=False, max_boxes=max_boxes)
            image_datas.append(image_data)
            boxes.append(box_data)
            i = (i + 1) % n

        image_datas = np.array(image_datas, dtype='float32')
        boxes = np.array(boxes, dtype='float32')
        y_trues = true_boxes_to_labels(boxes, input_shape, anchors, num_classes)
        yield [image_datas, *y_trues], np.zeros(batch_size)

