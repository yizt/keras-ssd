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
    num_classes = 2

    # image
    image_size = 300
    input_shape = (image_size, image_size, 3)
    # preprocess
    mean_pixel = [123.7, 116.8, 103.9]
    std = 127.5

    # gt boxes
    max_gt_num = 100
    #
    positive_iou_threshold = 0.5
    negative_iou_threshold = 0.4
    specs = [FeatureSpec(19, 16, 60, 105, [2, 1 / 2, 3, 1 / 3]),
             FeatureSpec(10, 30, 105, 150, [2, 1 / 2, 3, 1 / 3]),
             FeatureSpec(5, 60, 150, 195, [2, 1 / 2, 3, 1 / 3]),
             FeatureSpec(3, 100, 195, 240, [2, 1 / 2, 3, 1 / 3]),
             FeatureSpec(2, 150, 240, 285, [2, 1 / 2, 3, 1 / 3]),
             FeatureSpec(1, 300, 285, 330, [2, 1 / 2])]

    # detect boxes
    max_detections_per_class = 100
    max_total_detections = 100
    score_threshold = 0.5
    iou_threshold = 0.3

    # model
    base_model_name = 'resnet50'
    pretrained_weight_path = ''
    # 损失函数权重
    loss_weights = {
        "class_loss": 1.,
        "bbox_loss": 1.
    }

    @classmethod
    def feature_fn(cls, *args, **kwargs):
        return mobilenet_v2_features(*args, **kwargs)


class VocConfig(Config):
    num_classes = 1 + 20
    class_mapping = {'bg': 0,
                     'train': 1,
                     'dog': 2,
                     'bicycle': 3,
                     'bus': 4,
                     'car': 5,
                     'person': 6,
                     'bird': 7,
                     'chair': 8,
                     'diningtable': 9,
                     'sheep': 10,
                     'tvmonitor': 11,
                     'horse': 12,
                     'sofa': 13,
                     'bottle': 14,
                     'cat': 15,
                     'cow': 16,
                     'pottedplant': 17,
                     'boat': 18,
                     'motorbike': 19,
                     'aeroplane': 20
                     }
    voc_path = '/sdb/tmp/open_dataset/VOCdevkit'

    base_model_name = 'mobilenetv2'
    pretrained_weight_path = ''


cfg = VocConfig()
