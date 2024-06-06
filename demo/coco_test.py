"""
#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
@Project :pythonCodeSnippet
@File    :coco_test.py
@IDE     :PyCharm
@Author  :chenxw
@Date    :2024/1/5 18:17
@Descr:
"""

# 对coco_mask RLE编码的MASK进行还原

# polys mask rle三种格式互转
# https://blog.csdn.net/weixin_44966641/article/details/123171026?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-2-123171026-blog-104332577.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-2-123171026-blog-104332577.pc_relevant_default&utm_relevant_index=3

# RLE全称（run-length encoding），翻译为游程编码，又译行程长度编码，又称变动长度编码法（run coding），在控制论中对于二值图像而言是一种编码方法，对连续的黑、白像素数(游程)以不同的码字进行编码。游程编码是一种简单的非破坏性资料压缩法，其好处是加压缩和解压缩都非常快。其方法是计算连续出现的资料长度压缩之。
# RLE是COCO数据集的规范格式之一，也是许多图像分割比赛指定提交结果的格式。
# 返回值 rle 是一个字典，有两个字段 size 和 counts ，该字典通常直接作为 COCO 数据集的 segmentation 字段。

# RLE编码的理解推荐：https://blog.csdn.net/wuda19920215/article/details/113865418
# Ref：#
# https://wall.alphacoders.com/big.php?i=324547&lang=Chinese#
# https://blog.csdn.net/wuda19920215/article/details/113865418#
# https://www.cnblogs.com/aimhabo/p/9935815.html

import cv2
import numpy as np
import pycocotools.mask as mask_util

# polys 是一个 n × 2  的二维数组，表示多边形框的 n  个坐标，即 [ [ x 1 , y 1 ] , [ x 2 , y 2 ] , . . . [ x n , y n ] ] [[x_1,y_1],[x_2,y_2],...[x_n,y_n]]
polys = [[
    [
        329.0,
        289.0
    ],
    [
        327.0,
        289.0
    ],
    [
        326.0,
        290.0
    ],
    [
        325.0,
        290.0
    ],
    [
        322.0,
        293.0
    ],
    [
        321.0,
        293.0
    ],
    [
        321.0,
        296.0
    ],
    [
        320.0,
        297.0
    ],
    [
        320.0,
        298.0
    ],
    [
        319.0,
        299.0
    ],
    [
        319.0,
        301.0
    ],
    [
        317.0,
        303.0
    ],
    [
        317.0,
        304.0
    ],
    [
        316.0,
        305.0
    ],
    [
        316.0,
        309.0
    ],
    [
        322.0,
        309.0
    ],
    [
        323.0,
        308.0
    ],
    [
        324.0,
        308.0
    ],
    [
        325.0,
        307.0
    ],
    [
        326.0,
        308.0
    ],
    [
        327.0,
        308.0
    ],
    [
        329.0,
        310.0
    ],
    [
        330.0,
        310.0
    ],
    [
        331.0,
        309.0
    ],
    [
        331.0,
        308.0
    ],
    [
        332.0,
        307.0
    ],
    [
        332.0,
        306.0
    ],
    [
        333.0,
        305.0
    ],
    [
        333.0,
        300.0
    ],
    [
        334.0,
        299.0
    ],
    [
        334.0,
        296.0
    ],
    [
        333.0,
        295.0
    ],
    [
        333.0,
        294.0
    ],
    [
        330.0,
        291.0
    ],
    [
        330.0,
        290.0
    ]
]]
width = 800
height = 800
base_image_path = r"G:\data\test\airplane.png"
merge_image_path = r"G:\data\test\airplane_merge.png"
mask_image_path = r"G:\data\test\airplane_mask.png"


# rle(dict)转mask(ndarray)
def rle2mask(rle):
    mask = np.array(mask_util.decode(rle), dtype=np.float32)
    return mask


# rles(list)转masks(ndarray)
def rles2masks(rles):
    masks = []
    for rle in rles:
        mask = np.array(mask_util.decode(rle), dtype=np.float32)
        masks.append(mask)
    masks = np.array(masks)
    return masks


# mask(ndarray)转rle(dict)
def mask2rle(mask):
    # encoded with RLE
    rle = mask_util.encode(
        np.array(mask[:, :, np.newaxis], order='F',
                 dtype='uint8'))[0]
    if isinstance(rle['counts'], bytes):
        rle['counts'] = rle['counts'].decode()
    return rle


# mask(ndarray)转rle(dict)方法2
def mask2rle_2(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order='F', dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


# masks(ndarray)转rles(list)
def masks2rles(masks):
    rles = []
    for mask in masks:
        # encoded with RLE
        rle = mask_util.encode(
            np.array(mask[:, :, np.newaxis], order='F',
                     dtype='uint8'))[0]
        if isinstance(rle['counts'], bytes):
            rle['counts'] = rle['counts'].decode()
        rles.append(rle)

    return rles


# polys(list)装rles(list)
# polys:[[x1,y1,x2,y2...xn,yn],....]
def polys2rles(polys, width, height):
    rles = mask_util.frPyObjects(pyobj=polys, h=height, w=width)
    return rles


# labelme_polys:[[[x1,y1],[x2,y2]...[xn,yn]],...]
# coco_polys:[[x1,y1,x2,y2...xn,yn],....]
def labelme_polys2coco_polys(labelme_polys):
    coco_polys = []
    for labelme_poly in labelme_polys:
        # 二维数据打散到一维数据
        coco_poly = [num for sublist in labelme_poly for num in sublist]
        coco_polys.append(coco_poly)
    return coco_polys


# labelme_polys:[[[x1,y1],[x2,y2]...[xn,yn]],...]
# coco_polys:[[x1,y1,x2,y2...xn,yn],....]
def coco_polys2labelme_polys(coco_polys):
    labelme_polys = []
    for coco_poly in coco_polys:
        # 一维数据分到二维数据(每两个数字为一行)
        # 计算需要分组的次数
        group_count = len(coco_poly) // 2 + (len(coco_poly) % 2 != 0)
        # 利用列表切片进行分组并添加到新的二维列表中
        labelme_poly = [[coco_poly[i], coco_poly[i + 1]] for i in range(0, group_count * 2, 2)]
        labelme_polys.append(labelme_poly)
    return labelme_polys


# 在图上画polys
# polys:[[[x1,y1],[x2,y2]...[xn,yn]],[[x1,y1],[x2,y2]...[xn,yn]]...]
def draw_polys_on_image(base_image_path, polys, merge_image_path):
    img = cv2.imread(base_image_path)
    polys_points = np.array(polys, dtype=np.int32)
    cv2.polylines(img, polys_points, True, (255, 0, 0), 3)
    # 保存图像
    cv2.imwrite(merge_image_path, img)


# polys(list)转masks(ndarry)
def polys2masks(polys, width, height):
    masks = np.zeros((width, height), dtype=np.int32)
    obj = np.array(polys, dtype=np.int32)
    cv2.fillPoly(masks, obj, 1)
    return masks


# masks(ndarry)转polys(list)#
#  通过opencv的轮廓检测（等值线捕获）方法
# 返回[[[x1,y1],[x2,y2]...[xn,yn]],[...]],[[x1,y1],[x2,y2]...[xn,yn]],[...]],...]
def masks2polys(masks, tolerance=0.001):
    polygon_points = []
    # masks = cv2.imdecode(np.fromfile(mask_file, dtype=np.uint8), 0)

    masks = masks.astype(np.uint8)

    # 二值图边缘线，只能读单波段的灰度图
    contours, _ = cv2.findContours(masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        epsilon = tolerance * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 5:
            continue
        each_poly_points = []
        for point in approx:
            x, y = point[0].tolist()
            each_poly_points.append([x, y])
        polygon_points.append(each_poly_points)
    return polygon_points

    # def close_contour(contour):
    #     if not np.array_equal(contour[0], contour[-1]):
    #         contour = np.vstack((contour, contour[0]))
    #     return contour
    #
    # """Converts a binary mask to COCO polygon representation
    # Args:
    # binary_mask: a 2D binary numpy array where '1's represent the object
    # tolerance: Maximum distance from original points of polygon to approximated
    # polygonal chain. If tolerance is 0, the original coordinate array is returned.
    # """
    # polygons = []
    # # pad mask to close contours of shapes which start and end at an edge
    # padded_binary_mask = np.pad(masks, pad_width=1, mode='constant', constant_values=0)
    # contours = measure.find_contours(padded_binary_mask, 0.5)
    # contours = np.subtract(contours, 1)
    # for contour in contours:
    #     contour = close_contour(contour)
    #     contour = measure.approximate_polygon(contour, tolerance)
    #     if len(contour) < 3:
    #         continue
    #     contour = np.flip(contour, axis=1)
    #     segmentation = contour.ravel().tolist()
    #     # after padding and subtracting 1 we may get -0.5 points in our segmentation
    #     segmentation = [0 if i < 0 else i for i in segmentation]
    #     polygons.append(segmentation)
    # return polygons


# mask(ndarry)转poly(list)#
#  通过opencv的轮廓检测（等值线捕获）方法
# 返回[[x1,y1],[x2,y2]...[xn,yn]],[...]]]
def mask2poly(mask, tolerance=0.001):
    each_poly_points = []
    # masks = cv2.imdecode(np.fromfile(mask_file, dtype=np.uint8), 0)
    # mask=np.array([mask])
    mask = mask.astype(np.uint8)

    # 二值图边缘线，只能读单波段的灰度图
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    each_poly_points = []
    if len(contours) > 0:
        contour = contours[0]
        epsilon = tolerance * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) >= 5:
            for point in approx:
                x, y = point[0].tolist()
                each_poly_points.append([x, y])
    return each_poly_points


# masks数据(ndarry)转mask二值图
def masks2image(mask_image_path, mask_data):
    color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask_data.shape[-2:]
    # 转换为是三维数组,mask为True,采用样色值(0.11764706,0.56470588,1.0,0.6),mask为False，设置为（0,0,0,0）
    mask_image = mask_data.reshape(h, w, 1) * color.reshape(1, 1, -1)
    # 生成一个灰度图图片
    # 获取图片每个通道数据
    r, g, b, a = mask_image[:, :, 0], mask_image[:, :, 1], mask_image[:, :, 2], mask_image[:, :, 3]
    # 目标是255白色，背景是0黑色
    mask_image_grey = np.where(r > 0, 255, 0)
    # 保存mask的二值图
    cv2.imwrite(mask_image_path, mask_image_grey)


# mask二值图转polygon
# 通过opencv的轮廓检测（等值线捕获）方法
# 返回[[[x1,y1],[x2,y2]...[xn,yn]],[...]],[[x1,y1],[x2,y2]...[xn,yn]],[...]],...]
def maskimage_to_polygon(mask_file, epsilon_factor=0.001):
    polygon_points = []
    # binary_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    binary_mask = cv2.imdecode(np.fromfile(mask_file, dtype=np.uint8), 0)
    # 二值图边缘线，只能读单波段的灰度图
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 5:
            continue
        each_poly_points = []
        for point in approx:
            x, y = point[0].tolist()
            each_poly_points.append([x, y])
        polygon_points.append(each_poly_points)
    return polygon_points


print("---------原始的polys------")
print(polys)
draw_polys_on_image(base_image_path, polys, merge_image_path)
masks = polys2masks(polys, width, height)
print("---------转换后的masks------")
print(masks)
masks2image(mask_image_path, masks)
polys = masks2polys(masks, 0.001)
print("---------转换后的polys------")
print(polys)
polys2 = maskimage_to_polygon(mask_image_path, 0.001)
print("---------转换后的polys2------")
print(polys2)
polys3 = labelme_polys2coco_polys(polys)
print("---------转换后的polys3------")
print(polys3)
rles = polys2rles(polys3, width, height)
print("-----原始的rles----")
print(rles)
masks = rles2masks(rles)
print("-----转换的masks----")
print(masks)
rles = masks2rles(masks)
print("-----转换的rles----")
print(rles)

print("---------循环每个rle--------")
for rle in rles:
    print("-----原始的rle----")
    print(rle)
    mask = rle2mask(rle)
    print("-----转换的mask----")
    print(mask)
    rle = mask2rle(mask)
    print("-----转换的rle----")
    print(rle)

rle = {'size': [672, 1327], 'counts': 'dZW<7cd07O1O0O10000O0000000010O101O1N101O4LaQc>'}
print(rle)
mask = rle2mask(rle)
print("-----转换的mask----")
print(mask)
poly = mask2poly(mask)
print("-----转换的poly----")
print(poly)
rle = mask2rle(mask)
print("-----转换的rle----")
print(rle)
