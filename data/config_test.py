"""
#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
@Project :mmdetection_main
@File    :config_test.py
@IDE     :PyCharm
@Author  :chenxw
@Date    :2024/1/16 16:57
@Descr:
"""
from mmengine import Config
# 读取算法配置文件基类
cfg = Config.fromfile('mmdet-insseg/airplane/001_mask_rcnn_PLANE.py')
print(cfg.pretty_text)
