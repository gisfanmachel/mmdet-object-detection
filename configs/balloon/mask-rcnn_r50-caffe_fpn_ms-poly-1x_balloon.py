"""
#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
@Project :mmdetection-main
@File    :mask-rcnn_r50-caffe_fpn_ms-poly-1x_balloon.py
@IDE     :PyCharm
@Author  :chenxw
@Date    :2023/10/20 14:55
@Descr:
"""
# 新配置继承了基本配置，并做了必要的修改
_base_ = '../mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco.py'

# 我们还需要更改 head 中的 num_classes 以匹配数据集中的类别数
# 目标检测  box_head, 实例分割 mask_head
# 如果数据集太小，可以将backbone完全固定，backbone=dict(frozen_stages=4)
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1), mask_head=dict(num_classes=1)))

# 修改数据集相关配置
data_root = 'data/balloon/'
metainfo = {
    # 这里是元组，所有后面的逗号不能删除
    'classes': ('balloon', ),
    'palette': [
        (220, 20, 60),
    ]
}
# 定义日志和checkpoint保存的频次
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5,save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
# 定义训练的轮次
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=2, val_interval=1)
# 定义训练的批数据大小
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/annotation_coco.json',
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val/annotation_coco.json',
        data_prefix=dict(img='val/')))
test_dataloader = val_dataloader

# 修改评价指标相关配置
val_evaluator = dict(ann_file=data_root + 'val/annotation_coco.json')
test_evaluator = val_evaluator

# 使用预训练的 Mask R-CNN 模型权重来做初始化，可以提高模型性能
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
# 继续上次的训练，load_from 加载上次的pth