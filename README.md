### 检测图片 ###
检测图片执行以下脚本
```bash
python detect.py \
	--weights_path=model_data/yolov3.h5 \
	--source_images_dir=source_images \
	--classes_path=model_data/coco_classes.txt \
	--output_dir=detected_images
```