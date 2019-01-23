### 转换权重 ###
执行以下脚本，将`yolov3.weights`转换为`yolov3.h5`
```bash
python convert.py \
	--weights_path=yolov3.weights \
	--config_path=data/yolov3.cfg \
	--output_path=data/yolov3.h5
```

### 检测图片 ###
将需要检测的图片放到`samples`目录下，然后执行以下脚本
```bash
python detect.py \
	--weights_path=/home/easygo/yolov3/YOLOv3/log/trained_weights_stage_1.h5 \
	--source_images_dir=/home/easygo/yolov3/datasets/samples \
	--classes_path=/home/easygo/mc/keras-yolo3-master/model_data/coco_classes.txt \
	--output_dir=samples/detected
```

### 训练 ###

#### convert coco dataset ####
将coco数据集转成训练所需的目标格式，运行如下脚本
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

#### train ####
```
python train.py \
	--initial_weights_path=data/yolov3.h5 \
	--annotations_path=/home/easygo/mc/keras-yolo3-master/train.txt \
	--classes_path=/home/easygo/mc/keras-yolo3-master/model_data/coco_classes.txt \
	--use_focal_loss=0
```