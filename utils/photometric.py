# -*- coding: utf-8 -*-
"""
 @File    : photometric.py
 @Time    : 2019/10/27 下午3:57
 @Author  : yizuotian
 @Description    :
"""
import cv2


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
        return image, gt_boxes, labels

class Saturation:
    '''
    Changes the saturation of HSV images.

    Important:
        - Expects HSV input.
        - Expects input array to be of `dtype` `float`.
    '''

    def __init__(self, factor):
        '''
        Arguments:
            factor (float): A float greater than zero that determines saturation change, where
                values less than one result in less saturation and values greater than one result
                in more saturation.
        '''
        if factor <= 0.0: raise ValueError("It must be `factor > 0`.")
        self.factor = factor

    def __call__(self, image, labels=None):
        image[:, :, 1] = np.clip(image[:, :, 1] * self.factor, 0, 255)
        if labels is None:
            return image
        else:
            return image, labels