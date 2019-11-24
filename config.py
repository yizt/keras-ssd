# -*- coding: utf-8 -*-
"""
 @File    : config.py
 @Time    : 2019/11/24 下午2:38
 @Author  : yizuotian
 @Description    :
"""
from utils.anchor import FeatureSpec
from base_net.mobilenet import mobilenet_v2_features


class Config(object):
    num_classes = 20
    pretrained_weight_path = ''
    specs = [FeatureSpec(19, 16, 60, 105, [2, 1 / 2, 3, 1 / 3]),
             FeatureSpec(10, 30, 105, 150, [2, 1 / 2, 3, 1 / 3]),
             FeatureSpec(5, 60, 150, 195, [2, 1 / 2, 3, 1 / 3]),
             FeatureSpec(3, 100, 195, 240, [2, 1 / 2, 3, 1 / 3]),
             FeatureSpec(2, 150, 240, 285, [2, 1 / 2, 3, 1 / 3]),
             FeatureSpec(1, 300, 285, 330, [2, 1 / 2])]

    max_gt_num = 100
    #
    positive_iou_threshold = 0.5
    negative_iou_threshold = 0.4

    # detect boxes
    max_detections_per_class = 100
    max_total_detections = 100
    score_threshold = 0.5
    iou_threshold = 0.3

    @classmethod
    def feature_fn(cls, *args, **kwargs):
        return mobilenet_v2_features(*args, **kwargs)


cfg = Config()
