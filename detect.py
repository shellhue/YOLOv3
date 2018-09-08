from PIL import Image
import argparse
from YOLOv3 import YOLOv3
from os import walk
import os


parser = argparse.ArgumentParser(description='Darknet To Keras Converter.')
parser.add_argument('--weights_path', help='The weights used to initial model')
parser.add_argument('--source_images_dir', help='the dir of source images')
parser.add_argument('--classes_path', help='Path to classes.')
parser.add_argument('--output_dir', help='the dir to output detecting results.')


def _main(args):
    assert len(args.source_images_dir) > 0, "source images directory can not be empty!"
    assert len(args.output_dir) > 0, "detected image output directory can not be empty!"
    assert len(args.weights_path) > 0, "weights path can not be empty!"
    assert len(args.classes_path) > 0, "classes path can not be empty!"

    output_dir = args.output_dir if args.output_dir.endswith('/') else args.output_dir + '/'
    source_images_dir = args.source_images_dir if args.source_images_dir.endswith('/') else args.source_images_dir + '/'
    os.makedirs(output_dir, exist_ok=True)

    kwargs = {}
    if args.classes_path:
        kwargs['classes_path'] = args.classes_path
    if args.output_dir:
        kwargs['log_dir'] = args.output_dir

    yolo = YOLOv3(
        initial_weights_path=str(args.weights_path),
        is_training=False,
        **kwargs
    )

    source_images = []
    print('######### fetching all filenames ##########')
    for (_, _, filename) in walk(source_images_dir):
        source_images.extend(filename)

    for source_image_filename in source_images:
        try:
            print('######### loading image with name: {} ##########'.format(source_image_filename))
            image = Image.open(source_images_dir + source_image_filename)
        except:
            print('Invalid image file name: {}'.format(source_images_dir + source_image_filename))
            continue
        print('######### detecting image with name: {} ##########'.format(source_image_filename))
        detected_source_image = yolo.detect_image(image)
        print('######### saving detected image with name: {} ##########'.format(source_image_filename))
        detected_source_image.save(output_dir + source_image_filename, 'JPEG')
    print('######### finishing image detecting ##########')
    yolo.close_session()


if __name__ == '__main__':
    _main(parser.parse_args())
