# -*- coding: utf-8 -*-
"""
 @File    : anchor.py
 @Time    : 2019/11/14 下午2:59
 @Author  : yizuotian
 @Description    :
"""
from collections import namedtuple
from typing import List

import numpy as np

FeatureSpec = namedtuple('FeatureSpec',
                         ['feature_size', 'stride', 'min_size', 'max_size', 'aspect_ratios'])


def generate_anchors(specs: List[FeatureSpec], image_size):
    """

    :param specs:
       eg: specs = [
            FeatureSpec(19, 16, 60, 105, [2,1/2,3,1/3]),
            FeatureSpec(10, 32, 105, 150, [2,1/2,3,1/3]),
            FeatureSpec(5, 64, 150, 195, [2,1/2,3,1/3]),
            FeatureSpec(3, 100, 195, 240, [2,1/2,3,1/3]),
            FeatureSpec(2, 150, 240, 285, [2,1/2,3,1/3]),
            FeatureSpec(1, 300, 285, 330, [2,1/2,3,1/3])
        ]
    :param image_size:
    :return anchors: [N,(y1,x1,y2,x2)]
    """
    anchors = []
    for spec in specs:
        anchors.append(feature_map_anchors(spec.feature_size,
                                           spec.stride,
                                           spec.min_size,
                                           spec.max_size,
                                           spec.aspect_ratios))
    anchors = np.concatenate(anchors, axis=0)
    anchors = anchors / image_size
    return anchors


def feature_map_anchors(feature_size, stride, min_size, max_size, aspect_ratios):
    """
    某个feature map的anchor
    :param feature_size: feature map的尺寸
    :param stride: feature map的步长(相对图像的下采样倍数)
    :param min_size:
    :param max_size:
    :param aspect_ratios: 长宽比 shape:(M,)
    :return: （N*M,(y1,x1,y2,x2))
    """
    # anchor中心点坐标
    ys = xs = np.arange(0.5, feature_size, 1) * stride
    ctr_x, ctr_y = np.meshgrid(xs, ys)
    centers = np.stack([ctr_y, ctr_x], axis=-1)  # [H,W,(y,x)]
    centers = np.reshape(centers, (-1, 2))  # [N,(y,x)]

    # anchor长宽
    hw = list()
    hw.append([min_size, min_size])
    for r in aspect_ratios:
        hw.append([min_size * np.sqrt(r), min_size / np.sqrt(r)])
    hw.append([np.sqrt(min_size * max_size), np.sqrt(min_size * max_size)])

    hw = np.array(hw)
    # 相对anchor框中心点坐标
    coordinates = np.concatenate([-0.5 * hw, 0.5 * hw], axis=1)  # [M,(y1,x1,y2,x2)]

    # [N,1,4]+[1,M,4] => [N,M,4]
    anchors = centers[:, np.newaxis, [0, 1, 0, 1]] + coordinates[np.newaxis, :, :]

    return anchors.reshape((-1, 4))  # [N*M,4]


if __name__ == '__main__':
    # xs = np.arange(0.5, 3, 1)
    # ys = np.arange(0.5, 4, 1)
    # ctr_x, ctr_y = np.meshgrid(xs, ys)
    # ctrs = np.stack([ctr_y, ctr_x], axis=-1)
    # print(ctrs.shape)
    achrs = generate_anchors([FeatureSpec(19, 16, 60, 105, [2, 1 / 2, 3, 1 / 3]),
                              FeatureSpec(10, 30, 105, 150, [2, 1 / 2, 3, 1 / 3]),
                              FeatureSpec(5, 60, 150, 195, [2, 1 / 2, 3, 1 / 3]),
                              FeatureSpec(3, 100, 195, 240, [2, 1 / 2, 3, 1 / 3]),
                              FeatureSpec(2, 150, 240, 285, [2, 1 / 2, 3, 1 / 3]),
                              FeatureSpec(1, 300, 285, 330, [2, 1 / 2])],300)
    print(achrs)
    print(achrs.shape)
