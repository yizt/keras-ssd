# -*- coding: utf-8 -*-
"""
 @File    : augment.py
 @Time    : 2019/10/26 下午9:04
 @Author  : yizuotian
 @Description    : 目标检测图像数据增广
"""
import cv2
import numpy as np


class Resize(object):

    def __init__(self, height, width, interpolation_mode=cv2.INTER_LINEAR):
        """

        :param height:
        :param width:
        :param interpolation_mode: An integer that denotes a valid
                OpenCV interpolation mode. For example, integers 0 through 5 are
                valid interpolation modes.
        """
        self.out_height = height
        self.out_width = width
        self.interpolation_mode = interpolation_mode

    def __call__(self, image, gt_boxes=None):
        """

        :param image: [H,W,3]
        :param gt_boxes: GT boxes [N,(y1,x1,y2,x2)]
        :return image:
        :return boxes:
        """

        img_height, img_width = image.shape[:2]

        image = cv2.resize(image,
                           dsize=(self.out_width, self.out_height),
                           interpolation=self.interpolation_mode)
        if gt_boxes is None:
            return image

        boxes = gt_boxes.copy()
        boxes[:, [0, 2]] = np.round(boxes[:, [0, 2]] * (self.out_height / img_height), decimals=0)
        boxes[:, [1, 3]] = np.round(boxes[:, [1, 3]] * (self.out_width / img_width), decimals=0)
        return image, boxes


class Flip:
    """
    水平或者垂直翻转图像
    """

    def __init__(self, dim):
        """

        :param dim: 翻转维度；horizontal-水平，vertical-垂直
        """
        self.dim = dim

    def __call__(self, image, gt_boxes=None):
        """

        :param image: [H,W,3]
        :param gt_boxes: GT boxes [N,(y1,x1,y2,x2)]
        :return:
        """
        img_height, img_width = image.shape[:2]

        if self.dim == 'horizontal':  # 水平翻转
            image = image[:, ::-1]
            if gt_boxes is None:
                return image

            boxes = np.copy(gt_boxes)
            boxes[:, [1, 3]] = img_width - boxes[:, [3, 1]]
            return image, boxes
        else:  # 垂直翻转
            image = image[::-1]
            if gt_boxes is None:
                return image

            boxes = np.copy(gt_boxes)
            boxes[:, [0, 2]] = img_height - boxes[:, [2, 0]]
            return image, boxes


class HorizontalFlip(Flip):
    """
    水平翻转
    """

    def __init__(self):
        super(HorizontalFlip, self).__init__(dim='horizontal')


class VerticalFlip(Flip):
    """
    垂直翻转
    """

    def __init__(self):
        super(VerticalFlip, self).__init__(dim='vertical')
