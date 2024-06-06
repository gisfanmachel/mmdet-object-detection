"""
#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
@Project :mmdetection_main
@File    :tools.py
@IDE     :PyCharm
@Author  :chenxw
@Date    :2024/1/16 15:53
@Descr:
"""


def get_config_file_by_alg_model(alg_model):
    config_path = None
    if alg_model == "mmdet_v3-insseg-mask_rcnn_mask":
        config_path = "mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco.py"
    elif alg_model == "mmdet_v3-objdet-cascade_rcnn_hbb":
        config_path = "cascade_rcnn/cascade-rcnn_r101_fpn_1x_coco.py"
    elif alg_model == "mmdet_v3-insseg-cascade_mask_rcnn_mask":
        config_path = "cascade_rcnn/cascade-mask-rcnn_r50_fpn_ms-3x_coco.py"
    return config_path


def get_checkpoint_file_by_alg_model(alg_model, object_name):
    checkpoint_path = None
    dirname = alg_model.split("-")[0].split("_")[0] + "-" + alg_model.split("-")[1]
    if alg_model == "mmdet_v3-insseg-mask_rcnn_mask":
        checkpoint_path = "{}/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth".format(
            dirname)
    elif alg_model == "mmdet_v3-objdet-cascade_rcnn_hbb":
        checkpoint_path = "{}/cascade_rcnn_r101_fpn_1x_coco_20200317-0b6a2fbf.pth".format(
            dirname)
    elif alg_model == "mmdet_v3-insseg-cascade_mask_rcnn_mask":
        checkpoint_path = "{}/cascade_mask_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210707_002651-6e29b3a6.pth".format(
            dirname)
    return checkpoint_path


def get_workdir_by_alg_model(alg_model, object_name):
    workdir = None
    # mmdet-insseg-cascade_rcnn_mask_airplane
    alg_name = alg_model.split("-")[2]
    workdir = "{}_{}".format(alg_name, object_name)
    return workdir


def get_new_config_name_by_alg_model(alg_model, object_name):
    dirname = alg_model.split("-")[0].split("_")[0] + "-" + alg_model.split("-")[1]
    new_config_path = "{}/{}/{}.py".format(dirname, object_name, get_workdir_by_alg_model(alg_model, object_name))
    return new_config_path


def is_has_mask_result(alg_model):
    is_has_mask = False
    if alg_model.split("_")[-1] == "mask":
        is_has_mask = True
    return is_has_mask
