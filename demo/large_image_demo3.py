"""
#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
@Project :mmdetection_main
@File    :large_image_demo3.py
@IDE     :PyCharm
@Author  :chenxw
@Date    :2024/1/19 11:00
@Descr:
"""
import os

os.getcwd()

# arrange an instance segmentation model for test


from sahi import AutoDetectionModel
from sahi.predict import get_prediction

model_path = "../checkpoints/vgis/airplane/airplane_yolo_v8-objdet-yolo_hbb_v2.pt"
image_path = "../demo/airplane/bigimg.jpg"

# 初始化检测模型，缺少yolov5代码，pip install yolov5即可
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',  # 模型类型
    model_path=model_path,  # 模型文件路径
    confidence_threshold=0.3,  # 检测阈值
    device="cuda:0",  # or 'cuda:0'
);

# https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_mmdetection.ipynb#scrollTo=3TwBUULP7LRD
# detection_model = AutoDetectionModel.from_pretrained(
#     model_type='mmdet',
#     model_path=model_path,
#     config_path=config_path,
#     confidence_threshold=0.4,
#     image_size=640,
#     device="cpu", # or 'cuda:0'
# )

# # 对图像进行切片slices = sahi.slice_image(image, slice_size=(416,416))
#
# # 创建一个线程池pool = sahi.create_thread_pool()
#
# # 对每个切片进行并行推理results = pool.map(lambda slice: yolov8.detect_objects(model, slice), slices)
#
# # 关闭线程池pool.close()
#
# # 合并推理结果final_results = sahi.merge_results(results)
#
# # 打印最终检测结果print(final_results)

# 获得模型直接预测结果
result = get_prediction(image_path, detection_model)

# result是SAHI的PredictionResult对象，可获得推理时间，检测图像，检测图像尺寸，检测结果
# 查看标注框，可以用于保存为其他格式
for pred in result.object_prediction_list:
    bbox = pred.bbox  # 标注框BoundingBox对象，可以获得边界框的坐标、面积
    category = pred.category  # 类别Category对象，可获得类别id和类别名
    score = pred.score.value  # 预测置信度

# 保存文件结果
export_dir = "result"
file_name = "res"
result.export_visuals(export_dir=export_dir, file_name=file_name)

# 展示结果
from PIL import Image
import os

image_path = os.path.join(export_dir, file_name + '.png')
img = Image.open(image_path).convert('RGB')
img.show()
