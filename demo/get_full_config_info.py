"""
#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
@Project :mmdetection_main
@File    :get_full_config_info.py
@IDE     :PyCharm
@Author  :chenxw
@Date    :2024/1/26 14:16
@Descr:
"""
from mmengine import Config

# cfg = Config.fromfile('../configs/cascade_rcnn/cascade-rcnn_x101_64x4d_fpn_20e_coco.py')
cfg = Config.fromfile('../configs/faster_rcnn/faster-rcnn_x101-32x4d_fpn_ms-3x_coco.py')
print(cfg)