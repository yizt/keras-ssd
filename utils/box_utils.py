# -*- coding: utf-8 -*-
"""
 @File    : box_utils.py
 @Time    : 2019/10/27 上午9:45
 @Author  : yizuotian
 @Description    : 边框工具类
"""
import numpy as np


def iou_nvn(boxes_a, boxes_b):
    """
    numpy 计算IoU
    :param boxes_a: [N,4]
    :param boxes_b: [M,4]
    :return:  IoU [N,M]
    """
    # 扩维
    boxes_a = np.expand_dims(boxes_a, axis=1)  # (N,1,4)
    boxes_b = np.expand_dims(boxes_b, axis=0)  # (1,M,4)

    # 分别计算高度和宽度的交集
    overlap_h = np.maximum(0.0,
                           np.minimum(boxes_a[..., 2], boxes_b[..., 2]) -
                           np.maximum(boxes_a[..., 0], boxes_b[..., 0]))  # (N,M)

    overlap_w = np.maximum(0.0,
                           np.minimum(boxes_a[..., 3], boxes_b[..., 3]) -
                           np.maximum(boxes_a[..., 1], boxes_b[..., 1]))  # (N,M)

    # 交集
    overlap = overlap_w * overlap_h

    # 计算面积
    area_a = (boxes_a[..., 2] - boxes_a[..., 0]) * (boxes_a[..., 3] - boxes_a[..., 1])
    area_b = (boxes_b[..., 2] - boxes_b[..., 0]) * (boxes_b[..., 3] - boxes_b[..., 1])

    # 交并比
    return overlap / (area_a + area_b - overlap)


def iou_1vn(box, boxes):
    """
    numpy 计算IoU,一对多
    :param box: [4]
    :param boxes: [N,4]
    :return iou: [N]
    """
    iou = iou_nvn(box[np.newaxis, :], boxes)  # [1,N]
    return iou[0]
