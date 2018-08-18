from model.model import train


def _main():
    annotation_path = '/home/shellhue/data/train.txt'
    classes_path = './model_data/coco_classes.txt'
    anchors_path = './model_data/yolo_anchors.txt'
    weights_path = './model_data/yolo.h5'
    log_dir = './model_data/log'

    train(annotation_path, classes_path, anchors_path, weights_path, log_dir)


if __name__ == '__main__':
    _main()