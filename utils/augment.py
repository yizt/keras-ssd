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
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * (self.out_height / img_height)
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * (self.out_width / img_width)
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


class Translate:
    """
    平移图像
    """

    def __init__(self, dy, dx, clip_boxes=True, background=(0, 0, 0)):
        """
        
        :param dy: 高度方向移动因子
        :param dx: 宽度方向移动因子
        :param clip_boxes: 裁剪边框到图像内
        :param background: 填充色
        """

        self.dy_rel = dy
        self.dx_rel = dx
        self.clip_boxes = clip_boxes
        self.background = background

    def __call__(self, image, gt_boxes=None):
        """

        :param image: [H,W,3]
        :param gt_boxes: GT boxes [N,(y1,x1,y2,x2)]
        :return:
        """

        img_height, img_width = image.shape[:2]

        dy_abs = img_height * self.dy_rel
        dx_abs = img_width * self.dx_rel
        matrix = np.float32([[1, 0, dx_abs],
                             [0, 1, dy_abs]])

        # 平移图像
        image = cv2.warpAffine(image,
                               M=matrix,
                               dsize=(img_width, img_height),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=self.background)

        if gt_boxes is None:
            return image

        # 边框坐标对应平移
        boxes = np.copy(gt_boxes)
        boxes[:, [1, 3]] += dx_abs
        boxes[:, [0, 2]] += dy_abs

        if self.clip_boxes:
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], a_min=0, a_max=img_width)
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], a_min=0, a_max=img_height)

        return image, boxes


class Scale:
    """
    缩放图像,图像尺寸不变,镜头拉近、拉远
    """

    def __init__(self,
                 factor,
                 clip_boxes=True,
                 background=(0, 0, 0)):
        """

        :param factor: 大于1拉近距离放大图像;小于1拉远距离缩小图像
        :param clip_boxes: 裁剪边框到图像内
        :param background:
        """

        self.factor = factor
        self.clip_boxes = clip_boxes
        self.background = background

    def __call__(self, image, gt_boxes=None):
        """

        :param image: [H,W,3]
        :param gt_boxes: GT boxes [N,(y1,x1,y2,x2)]
        :return:
        """

        img_height, img_width = image.shape[:2]

        matrix = cv2.getRotationMatrix2D(center=(img_width / 2, img_height / 2),
                                         angle=0,
                                         scale=self.factor)
        # 缩放图像
        image = cv2.warpAffine(image,
                               M=matrix,
                               dsize=(img_width, img_height),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=self.background)

        if gt_boxes is None:
            return image

        boxes = gt_boxes.copy()

        # 计算缩放后，左上右下两点坐标  [2,3]*[(x,y,1),N]=>[(x,y),N]
        top_left = np.array([boxes[:, 1], boxes[:, 0], np.ones(boxes.shape[0])])  # [(x,y,1),N]
        bottom_right = np.array([boxes[:, 3], boxes[:, 2], np.ones(boxes.shape[0])])  # [(x,y,1),N]
        new_top_left = (np.dot(matrix, top_left)).T  # [N,(x,y)]
        new_bottom_right = (np.dot(matrix, bottom_right)).T  # [N,(x,y)]

        boxes[:, [1, 0]] = new_top_left
        boxes[:, [3, 2]] = new_bottom_right

        if self.clip_boxes:
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], a_min=0, a_max=img_width)
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], a_min=0, a_max=img_height)

        return image, boxes
