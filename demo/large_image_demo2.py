# Copyright (c) OpenMMLab. All rights reserved.
"""Perform MMDET inference on large images (as satellite imagery) as:

```shell
wget -P checkpoint https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_2x_coco/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth # noqa: E501, E261.

python demo/large_image_demo.py \
    demo/large_image.jpg \
    configs/faster_rcnn/faster-rcnn_r101_fpn_2x_coco.py \
    checkpoint/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth
```
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from sahi.models.mmdet import MmdetDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image, visualize_object_predictions
from IPython.display import Image

try:
    from sahi.slicing import slice_image
except ImportError:
    raise ImportError('Please run "pip install -U sahi" '
                      'to install sahi first for large image inference.')

import platform
platformType = platform.system().lower()


if __name__ == '__main__':
    # if platformType == "windows":
    #     model_path = r"E:\系统开发\AI\AI_Study\mmdetection_main\checkpoints\vgis\airplane\airplane_mmdet_v3-insseg-mask_rcnn_mask.pth"
    #     config_path = r"E:\系统开发\AI\AI_Study\mmdetection_main\configs\vgis\airplane\airplane_mmdet_v3-insseg-mask_rcnn_mask.py"
    #     image_path = r"E:\系统开发\AI\AI_Study\mmdetection_main\demo\airplane\bigimg.jpg"
    #
    #
    #     model_path = "../checkpoints/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth"
    #     config_path = "../configs/faster_rcnn/faster-rcnn_r101_fpn_2x_coco.py"
    #     image_path = "../demo/large_image.jpg"
    # else:
    #     model_path = "../checkpoints/vgis/airplane/airplane_mmdet_v3-insseg-mask_rcnn_mask.pth"
    #     config_path = "../configs/vgis/airplane/airplane_mmdet_v3-insseg-mask_rcnn_mask.py"
    #     image_path = "../demo/airplane/bigimg.jpg"


    model_path = "../checkpoints/vgis/airplane/airplane_mmdet_v3-insseg-mask_rcnn_mask.pth"
    config_path = "../data/mmdet-insseg/airplane/airplane_mmdet_v3-insseg-mask_rcnn_mask.py"
    image_path = "../demo/airplane/bigimg.jpg"

    # 需要做并行，效率没有large_image_demo.py 高

    # model_path = "../checkpoints/vgis/boat/boat_mmdet_v3_insseg-mask-rcnn_mask.pth"
    # config_path = "../data/mmdet-insseg/boat/boat_mmdet_v3-insseg-mask_rcnn_mask.py"
    # image_path = "../demo/boat/big_img.jpg"

    detection_model = MmdetDetectionModel(
        model_path=model_path,
        config_path=config_path,
        mask_threshold=0.5,
        confidence_threshold=0.3,
        device="cuda:0"
    )

    image = read_image(image_path)

    result = get_sliced_prediction(
        image,
        detection_model,
        slice_height=2560,
        slice_width=2560,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    # 对图像进行切片slices = sahi.slice_image(image, slice_size=(416,416))

    # 创建一个线程池pool = sahi.create_thread_pool()

    # 对每个切片进行并行推理results = pool.map(lambda slice: yolov8.detect_objects(model, slice), slices)

    # 关闭线程池pool.close()

    # 合并推理结果final_results = sahi.merge_results(results)

    # 打印最终检测结果print(final_results)


    result.export_visuals(export_dir="demo_data/",file_name= "prediction_visual2",text_size=2)

    # Image("demo_data/prediction_visual.png")
    #
    #
    # print(len(result.object_prediction_list))
    # print(result.object_prediction_list[0].bbox)
    #
    # visualization_result = visualize_object_predictions(
    #     image,
    #     object_prediction_list=result.object_prediction_list,
    #     output_dir="",
    #     file_name="",
    # )
    # print(visualization_result["image"])

    # result.object_prediction_list[0].bbox.miny    #
    #
    #
    # result.object_prediction_list[0].category.id/name
    #
    #
    # result.object_prediction_list[0].score.value
    #
    #
    # result.object_prediction_list[0].mask.full_shape_height
    #
    #
    # result.object_prediction_list[0].mask.full_shape_width
    #
    #
    #
    # result.object_prediction_list[0].mask.bool_mask.shape    #
    #
    #
    # result.object_prediction_list[0].mask.bool_mask   ndarray类型