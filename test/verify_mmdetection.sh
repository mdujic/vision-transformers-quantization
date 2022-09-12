mkdir -p /tmp/verify_mmdetection_workdir
mim download mmdet --config faster_rcnn_r50_fpn_1x_coco --dest /tmp/verify_mmdetection_workdir
python /vt-internship/test/verify_mmdetection.py