# training schedule for 20e

max_epochs = 100
base_lr = 16 * 0.004 / (32 * 8)

# 修改max_epochs,val_interval
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=20,
        by_epoch=True,
        milestones=[16, 19],
        gamma=0.1)
]

# optimizer
# 修改学习率lr
# # batch 改变了，学习率也要跟着改变， 0.004 是 8卡x32 的学习率
# lr = 16 * 0.004 / (32 * 8)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=base_lr, momentum=0.9, weight_decay=0.0001))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
