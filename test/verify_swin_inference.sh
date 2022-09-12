mkdir -p /tmp/verify_swin_inference_workdir
mim download mmdet --config mask_rcnn_swin-t-p4-w7_fpn_1x_coco --dest /tmp/verify_swin_inference_workdir
python /vt-internship/test/verify_swin_inference.py