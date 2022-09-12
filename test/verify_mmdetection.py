from mmdet.apis import init_detector, inference_detector


def main():
    config_file = '/tmp/verify_mmdetection_workdir/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = '/tmp/verify_mmdetection_workdir/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    image_file = '/mmdetection/demo/demo.jpg'
    device = 'cuda:0'
    # init a detector
    model = init_detector(config_file, checkpoint_file, device=device)
    # inference the demo image
    results = inference_detector(model, image_file)
    print(results)


if __name__ == "__main__":
    main()
