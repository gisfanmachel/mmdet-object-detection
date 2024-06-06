"""
#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
@Project :mmdetection_main
@File    :build_config.py
@IDE     :PyCharm
@Author  :chenxw
@Date    :2024/1/16 14:25
@Descr:
"""
from mmengine import Config

from config_tools import get_config_file_by_alg_model, get_checkpoint_file_by_alg_model, get_workdir_by_alg_model, \
    is_has_mask_result, get_new_config_name_by_alg_model

# alg_model = "mmdet_v3-insseg-mask_rcnn_mask"
alg_model = "mmdet_v3-objdet-cascade_rcnn_hbb"
alg_model = "mmdet_v3-insseg-cascade_mask_rcnn_mask"
object_name = "airplane"

# 读取算法配置文件基类
cfg = Config.fromfile('../configs/{}'.format(get_config_file_by_alg_model(alg_model)))
# 定义数据目录
data_root = './data/mmdet-insseg/airplane'
# 定义数据类别和调色板
metainfo = {
    'classes': ('transportplane', 'fighter', 'helicopter', 'transport'),
    'palette': [(101, 205, 228), (240, 128, 128), (154, 205, 50), (34, 139, 34)]
}
# 定义数据分类数量
num_classes = len(metainfo["classes"])
# 定义训练的epochs
max_epochs = 100
# 定义训练数据(单卡)的batch size
train_batch_size_per_gpu = 16
# 加载训练数据(batch)的线程数
train_num_workers = 4
# 定义验证数据(单卡)的batch size
val_batch_size_per_gpu = 1
# 加载验证数据(batch)的线程数
val_num_workers = 2
# 根据训练batch数调整训练的学习率， 原来的0.004 是 8卡x32 的学习率
base_lr = train_batch_size_per_gpu * 0.004 / (32 * 8)
# 定义要采用的COCO预训练权重
load_from = './data/{}'.format(get_checkpoint_file_by_alg_model(alg_model, object_name))  # noqa

# 设置输出的工作目录
work_dirs = '/root/work/mmdetection-main/work_dirs/{}'.format(get_workdir_by_alg_model(alg_model, object_name))

# Modify dataset classes and color
cfg.metainfo = metainfo

# Modify dataset type and path
cfg.data_root = data_root

cfg.train_dataloader.batch_size = train_batch_size_per_gpu
cfg.train_dataloader.num_workers = train_num_workers
cfg.train_dataloader.pin_memory = False
cfg.train_dataloader.dataset.ann_file = 'train/annotation_coco.json'
cfg.train_dataloader.dataset.data_root = cfg.data_root
cfg.train_dataloader.dataset.data_prefix.img = 'train/'
cfg.train_dataloader.dataset.metainfo = cfg.metainfo

cfg.val_dataloader.batch_size = val_batch_size_per_gpu
cfg.val_dataloader.num_workers = val_num_workers
cfg.val_dataloader.dataset.ann_file = 'val/annotation_coco.json'
cfg.val_dataloader.dataset.data_root = cfg.data_root
cfg.val_dataloader.dataset.data_prefix.img = 'val/'
cfg.val_dataloader.dataset.metainfo = cfg.metainfo

cfg.test_dataloader = cfg.val_dataloader

# Modify metric config
cfg.val_evaluator.ann_file = cfg.data_root + '/' + 'val/annotation_coco.json'
cfg.test_evaluator = cfg.val_evaluator

# box框的分类数
if isinstance(cfg.model.roi_head.bbox_head, dict):
    cfg.model.roi_head.bbox_head.num_classes = num_classes
elif isinstance(cfg.model.roi_head.bbox_head, list):
    for bbox_head in cfg.model.roi_head.bbox_head:
        bbox_head.num_classes = num_classes

if is_has_mask_result(alg_model):
    # mask的分类数
    if isinstance(cfg.model.roi_head.mask_head, dict):
        cfg.model.roi_head.mask_head.num_classes = num_classes
    elif isinstance(cfg.model.roi_head.mask_head, list):
        for mask_head in cfg.model.roi_head.mask_head:
            mask_head.num_classes = num_classes

# We can still the pre-trained Mask RCNN model to obtain a higher performance
cfg.load_from = load_from

# Set up working dir to save files and logs.
cfg.work_dir = work_dirs

# We can set the evaluation interval to reduce the evaluation times
cfg.train_cfg.val_interval = 5
cfg.train_cfg.max_epochs = max_epochs

# We can set the checkpoint saving interval to reduce the storage cost
cfg.default_hooks.checkpoint.interval = 5
# cfg.default_hooks.checkpoint.max_keep_ckpts = 2
# cfg.default_hooks.checkpoint.save_best = 'auto'

# cfg.default_hooks.logger.type = 'LoggerHook'
cfg.default_hooks.logger.interval = 5

# cfg.param_scheduler[0].type = 'LinearLR'
# cfg.param_scheduler[0].start_factor = 1.0e-5
# cfg.param_scheduler[0].by_epoch = False
# cfg.param_scheduler[0].begin = 0
# cfg.param_scheduler[0].end = 1000
#
# cfg.param_scheduler[1].type = 'CosineAnnealingLR'
# cfg.param_scheduler[1].eta_min = base_lr * 0.05
# cfg.param_scheduler[1].begin = max_epochs // 2
# cfg.param_scheduler[1].end = max_epochs
# cfg.param_scheduler[1].T_max = max_epochs // 2
# cfg.param_scheduler[1].by_epoch = True
# cfg.param_scheduler[1].convert_to_iter_based = True


# param_scheduler = [
#     dict(
#         type='LinearLR',  # 使用线性学习率预热
#         start_factor=0.001, # 学习率预热的系数
#         by_epoch=False,  # 按 iteration 更新预热学习率
#         begin=0,  # 从第一个 iteration 开始
#         end=500),  # 到第 500 个 iteration 结束
#     dict(
#         type='MultiStepLR',  # 在训练过程中使用 multi step 学习率策略
#         by_epoch=True,  # 按 epoch 更新学习率
#         begin=0,   # 从第一个 epoch 开始
#         end=12,  # 到第 12 个 epoch 结束
#         milestones=[8, 11],  # 在哪几个 epoch 进行学习率衰减
#         gamma=0.1)  # 学习率衰减系数
# ]

# 之前训练的报错，缺少type，是指这个地方，没有写 , type='OptimWrapper'
cfg.optim_wrapper.optimizer.lr = base_lr
# cfg.optim_wrapper = dict(optimizer=dict(lr=base_lr, momentum=0.9, type='SGD', weight_decay=0.0001), type='OptimWrapper')

# # Change the evaluation metric since we use customized dataset.
# cfg.evaluation.metric = 'mAP'
# # We can set the evaluation interval to reduce the evaluation times
# cfg.evaluation.interval = 12
# # We can set the checkpoint saving interval to reduce the storage cost
# cfg.checkpoint_config.interval = 12
#
# # Set seed thus the results are more reproducible
# cfg.seed = 0
# set_random_seed(0, deterministic=False)
# cfg.gpu_ids = range(1)
#
#
# # We can also use tensorboard to log the training process
# cfg.visualizer.vis_backends.append({"type":'TensorboardVisBackend'})

# ------------------------------------------------------


config = f'./' + get_new_config_name_by_alg_model(alg_model, object_name)
with open(config, 'w') as f:
    f.write(cfg.pretty_text)
