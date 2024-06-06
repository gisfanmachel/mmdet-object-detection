_base_ = [
    './models/cascade-rcnn_r101_fpn_airplane.py',
    './datasets/coco_detection_vgis_airplane.py',
    '../schedules/schedule_vgis.py', '../runtime/vgis_runtime_cascade_rcnn.py'
]
