import json
from collections import defaultdict
import argparse
from os import path

parser = argparse.ArgumentParser(description='convert coco dataset to annotation .')
parser.add_argument('--coco_image_dir',
                    help='the image directory')
parser.add_argument('--coco_annotation_path', help='Path to annotations.')
parser.add_argument('--output_path', help='output path.')

def _main(args):
    name_box_id = defaultdict(list)
    f = open(args.coco_annotation_path, encoding='utf-8')
    data = json.load(f)

    annotations = data['annotations']
    for ant in annotations:
        id = ant['image_id']
        name = path.join(args.coco_image_dir, '%012d.jpg' % id)
        cat = ant['category_id']

        if cat >= 1 and cat <= 11:
            cat = cat - 1
        elif cat >= 13 and cat <= 25:
            cat = cat - 2
        elif cat >= 27 and cat <= 28:
            cat = cat - 3
        elif cat >= 31 and cat <= 44:
            cat = cat - 5
        elif cat >= 46 and cat <= 65:
            cat = cat - 6
        elif cat == 67:
            cat = cat - 7
        elif cat == 70:
            cat = cat - 9
        elif cat >= 72 and cat <= 82:
            cat = cat - 10
        elif cat >= 84 and cat <= 90:
            cat = cat - 11

        name_box_id[name].append([ant['bbox'], cat])
    f = open(args.output_path, 'w')
    for key in name_box_id.keys():
        f.write(key)
        box_infos = name_box_id[key]
        for info in box_infos:
            x_min = int(info[0][0])
            y_min = int(info[0][1])
            x_max = x_min + int(info[0][2])
            y_max = y_min + int(info[0][3])

            box_info = " %d,%d,%d,%d,%d" % (
                x_min, y_min, x_max, y_max, int(info[1]))
            f.write(box_info)
        f.write('\n')
    f.close()

if __name__ == '__main__':
    _main(parser.parse_args())
