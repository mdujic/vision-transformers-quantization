from mmdet.apis import init_detector, inference_detector


def main():
    config_file = '/tmp/verify_swin_inference_workdir/mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py'
    checkpoint_file = '/tmp/verify_swin_inference_workdir/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth'
    image_file = '/mmdetection/demo/demo.jpg'
    device = 'cuda:0'
    # init a detector
    model = init_detector(config_file, checkpoint_file, device=device)
    # inference the demo image
    results = inference_detector(model, image_file)
    print(results)


if __name__ == "__main__":
    main()
