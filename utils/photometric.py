# -*- coding: utf-8 -*-
"""
 @File    : photometric.py
 @Time    : 2019/10/27 下午3:57
 @Author  : yizuotian
 @Description    :
"""
import cv2
import numpy as np
import random


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
        image = image.astype(np.float32)
        image[:, :, 1] = np.clip(image[:, :, 1] * self.factor, 0, 255)
        return Identity()(image.astype(np.uint8), gt_boxes, labels)


class RandomSaturation(object):
    def __init__(self, prob=0.5, lower=0.5, upper=1.5):
        self.prob = prob
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "saturation upper must be >= lower."
        assert self.lower >= 0, "saturation lower must be non-negative."

    def __call__(self, image, gt_boxes=None, labels=None):
        if random.random() < self.prob:
            alpha = random.uniform(self.lower, self.upper)
            return Saturation(factor=alpha)(image, gt_boxes, labels)
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
        image = image.astype(np.float32)
        image = np.clip(image + self.delta, 0, 255)
        return Identity()(image.astype(np.uint8), gt_boxes, labels)


class RandomBrightness(object):
    def __init__(self, prob=0.5, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.prob = prob
        self.delta = delta

    def __call__(self, image, gt_boxes=None, labels=None):
        if random.random() < self.prob:
            delta = random.uniform(-self.delta, self.delta)
            return Brightness(delta)(image, gt_boxes, labels)
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
        image = image.astype(np.float32)
        image = np.clip(127.5 + self.factor * (image - 127.5), 0, 255)
        return Identity()(image.astype(np.uint8), gt_boxes, labels)


class RandomContrast(object):
    def __init__(self, prob=0.5, lower=0.5, upper=1.5):
        self.prob = prob
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, gt_boxes=None, labels=None):
        if random.random() < self.prob:
            alpha = random.uniform(self.lower, self.upper)
            return Contrast(factor=alpha)(image, gt_boxes, labels)
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
        image = image.astype(np.float32)
        image[:, :, 0] = np.clip(image[:, :, 0] + self.delta, 0, 180)
        return Identity()(image.astype(np.uint8), gt_boxes, labels)


class RandomHue(object):
    def __init__(self, prob=0.5, delta=18.0):
        assert -180 <= delta <= 180, "`delta` must be in the closed interval `[-180, 180]`."
        self.delta = delta
        self.prob = prob

    def __call__(self, image, gt_boxes=None, labels=None):
        if random.random() < self.prob:
            delta = random.uniform(-self.delta, self.delta)
            return Hue(delta)(image, gt_boxes, labels)
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


class ChannelSwap:
    """
    通道交换
    """

    def __init__(self, order):
        self.order = order

    def __call__(self, image, gt_boxes=None, labels=None):
        image = image[:, :, self.order]
        return Identity()(image, gt_boxes, labels)


class RandomChannelSwap(object):
    def __init__(self, prob=0.5):
        self.prob = prob
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, gt_boxes=None, labels=None):
        if random.random() < self.prob:
            order = self.perms[np.random.randint(0, len(self.perms))]
            return ChannelSwap(order)(image, gt_boxes, labels)
        return Identity()(image, gt_boxes, labels)
