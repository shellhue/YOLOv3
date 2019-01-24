# YOLOv3

## Introduction
YOLOv3 is implemented with keras backed by tensorflow.

YOLOv3 has been tested with environment:

- Python 3.6.5
- Keras 2.2.4
- tensorflow 1.10

---

## Quick Start
1. Download YOLOv3 weights from YOLO website.
2. Convert the Darknet YOLO model to a Keras model.
3. Put your images to `samples` directory
4. Run YOLO detection.

```bash
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py \
	--weights_path=yolov3.weights \
	--config_path=data/yolov3.cfg \
	--output_path=data/yolov3.h5
python detect.py \
	--weights_path=data/yolov3.h5 \
	--source_images_dir=samples \
	--classes_path=data/coco_classes.txt \
	--output_dir=samples/detected
```

## Training
1. Convert your dataset to the specified format
2. Run YOLO train


```
python coco_annotation.py \
	--coco_image_dir=/home/shellhue/coco2017/train2017 \
	--coco_annotation_path=/home/shellhue/coco2017/annotations/instances_train2017.json \
	--output_path=/home/shellhue/coco2017/train.txt

python coco_annotation.py \
	--coco_image_dir=/home/shellhue/coco2017/val2017 \
	--coco_annotation_path=/home/shellhue/coco2017/annotations/instances_val2017.json \
	--output_path=/home/shellhue/coco2017/val.txt
```

```
python train.py \
	--initial_weights_path=data/yolov3.h5 \
	--annotations_path=/home/easygo/mc/keras-yolo3-master/train.txt \
	--classes_path=/home/easygo/mc/keras-yolo3-master/model_data/coco_classes.txt \
	--use_focal_loss=0
```