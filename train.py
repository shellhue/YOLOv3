import argparse
from model.yolov3 import YOLOv3


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Darknet To Keras Converter.')
parser.add_argument('--initial_weights_path', help='The weights used to initial model')
parser.add_argument('--annotations_path', help='Path to annotations.')
parser.add_argument('--classes_path', help='Path to classes.')
parser.add_argument('--log_dir', help='the dir to log.')
parser.add_argument('--use_focal_loss', type=str2bool, help='Whether use focal loss.')


def _main(args):
    kwargs = {}
    if args.classes_path:
        kwargs['classes_path'] = args.classes_path
    if args.log_dir:
        kwargs['log_dir'] = args.log_dir

    yolo = YOLOv3(
        initial_weights_path=str(args.initial_weights_path),
        annotations_path=str(args.annotations_path),
        is_training=True,
        **kwargs
    )

    yolo.train(use_focal_loss=args.use_focal_loss)


if __name__ == '__main__':
    _main(parser.parse_args())
