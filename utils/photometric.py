# -*- coding: utf-8 -*-
"""
 @File    : photometric.py
 @Time    : 2019/10/27 下午3:57
 @Author  : yizuotian
 @Description    :
"""
import cv2
import numpy as np


class Identity(object):
    """
    恒等转换
    """

    def __init__(self):
        super(Identity, self).__init__()

    def __call__(self, image, gt_boxes=None, labels=None):
        if gt_boxes is None:
            return image
        return image, gt_boxes, labels


class ConvertColor(object):
    """
    色彩空间转换
    """

    def __init__(self, current, transform):
        self.transform = transform
        self.current = current

    def __call__(self, image, gt_boxes=None, labels=None):
        """

        :param image: [H,W,3]
        :param gt_boxes: [N,(y1,x1,y2,x2)]
        :param labels: [N]
        :return:
        """
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'RGB' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'BGR' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.current == 'HSV' and self.transform == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return Identity()(image, gt_boxes, labels)


class Saturation(object):
    """
    改变HSV图像的饱和度
    """

    def __init__(self, factor):
        if factor <= 0.0:
            raise ValueError("It must be `factor > 0`.")
        self.factor = factor

    def __call__(self, image, gt_boxes=None, labels=None):
        image[:, :, 1] = np.clip(image[:, :, 1] * self.factor, 0, 255)
        return Identity()(image, gt_boxes, labels)


class Brightness(object):
    """
    改变RGB图像的亮度
    """

    def __init__(self, delta):
        """
        :param delta: 改变的像素值
        """
        self.delta = delta

    def __call__(self, image, gt_boxes=None, labels=None):
        image = np.clip(image + self.delta, 0, 255)
        return Identity()(image, gt_boxes, labels)


class Contrast(object):
    """
    改变对比度
    """

    def __init__(self, factor):
        if factor <= 0.0:
            raise ValueError("It must be `factor > 0`.")
        self.factor = factor

    def __call__(self, image, gt_boxes=None, labels=None):
        image = np.clip(127.5 + self.factor * (image - 127.5), 0, 255)
        return Identity()(image, gt_boxes, labels)


class Hue(object):
    """
    改变颜色
    """

    def __init__(self, delta):
        if not (-180 <= delta <= 180):
            raise ValueError("`delta` must be in the closed interval `[-180, 180]`.")
        self.delta = delta

    def __call__(self, image, gt_boxes=None, labels=None):
        image[:, :, 0] = np.clip(image[:, :, 0] + self.delta, 0, 180)
        return Identity()(image, gt_boxes, labels)


class Gamma(object):
    """
    改变RGB图像的gamma值
    """

    def __init__(self, gamma):
        if gamma <= 0.0:
            raise ValueError("It must be `gamma > 0`.")
        self.gamma = gamma
        self.gamma_inv = 1.0 / gamma
        self.table = np.array([((i / 255.0) ** self.gamma_inv) * 255 for i in np.arange(0, 256)]).astype("uint8")

    def __call__(self, image, gt_boxes=None, labels=None):
        image = cv2.LUT(image, self.table)
        return Identity()(image, gt_boxes, labels)


class HistogramEqualization:
    """
    对HSV图像做直方图均衡化
    """

    def __init__(self):
        super(HistogramEqualization, self).__init__()

    def __call__(self, image, gt_boxes=None, labels=None):
        image[:, :, 2] = cv2.equalizeHist(image[:, :, 2])
        return Identity()(image, gt_boxes, labels)
