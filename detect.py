from PIL import Image
import argparse
from os import walk
import os
from model.yolov3 import YOLOv3

parser = argparse.ArgumentParser(description='Darknet To Keras Converter.')
parser.add_argument('--weights_path', help='The weights used to initial model')
parser.add_argument('--source_images_dir', help='the dir of source images')
parser.add_argument('--classes_path', help='the dir to classes path.')
parser.add_argument('--output_dir', help='the dir to output detecting results.')


def _main(args):
    assert len(args.source_images_dir) > 0, "source images directory can not be empty!"
    assert len(args.output_dir) > 0, "detected image output directory can not be empty!"
    assert len(args.weights_path) > 0, "weights path can not be empty!"

    output_dir = args.output_dir if args.output_dir.endswith('/') else args.output_dir + '/'
    source_images_dir = args.source_images_dir if args.source_images_dir.endswith('/') else args.source_images_dir + '/'
    os.makedirs(output_dir, exist_ok=True)
    classes_path = args.classes_path if args.classes_path else 'data/coco_classes.txt'

    kwargs = {}

    yolo = YOLOv3(
        initial_weights_path=str(args.weights_path),
        is_training=False,
        anchors_path='data/yolo_anchors.txt',
        classes_path=classes_path,
        log_dir='log',
        **kwargs
    )

    source_images = []
    print(source_images_dir)
    print('######### fetching all filenames ##########')
    for (_, _, filename) in walk(source_images_dir):
        files = []
        for f in filename:
            if f.lower().endswith('.jpg') or f.lower().endswith('.png') or f.lower().endswith('.jpeg'):
                files.append(source_images_dir + f)
        source_images.extend(files)
    print(source_images)
    for source_image_path in source_images:
        try:
            image = Image.open(source_image_path)
        except:
            continue
        print('######### detecting image with name: {} ##########'.format(
            source_image_path))
        detected_source_image = yolo.detect_image(image)
        print('######### detected image saved to: {} ##########'.format(
            output_dir + source_image_path.split('/')[-1]))
        detected_source_image.save(output_dir + source_image_path.split('/')[-1], 'JPEG')
    print('######### finishing image detecting ##########')
    yolo.close_session()


if __name__ == '__main__':
    _main(parser.parse_args())
