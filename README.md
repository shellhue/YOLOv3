### 转换权重 ###
执行以下脚本，将`yolov3.weights`转换为`yolov3.h5`
```bash
python convert.py \
	--weights_path=yolov3.weights \
	--config_path=data/yolov3.cfg \
	--output_path=data/yolov3.h5
```

### 检测图片 ###
检测图片执行以下脚本
```bash
python detect.py \
	--weights_path=data/yolov3.h5 \
	--source_images_dir=samples \
	--classes_path=data/coco_classes.txt \
	--output_dir=samples/detected
```