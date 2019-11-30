# -*- coding: utf-8 -*-
"""
 @File    : config.py
 @Time    : 2019/11/24 下午2:38
 @Author  : yizuotian
 @Description    :
"""
from base_net import mobilenet, densenet
from utils.anchor import FeatureSpec


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
    # anchors,正负样本
    positive_iou_threshold = 0.5
    negative_iou_threshold = 0.4
    # specs = [FeatureSpec(19, 16, 60, 105, [2, 1 / 2, 3, 1 / 3]),
    #          FeatureSpec(10, 30, 105, 150, [2, 1 / 2, 3, 1 / 3]),
    #          FeatureSpec(5, 60, 150, 195, [2, 1 / 2, 3, 1 / 3]),
    #          FeatureSpec(3, 100, 195, 240, [2, 1 / 2, 3, 1 / 3]),
    #          FeatureSpec(2, 150, 240, 285, [2, 1 / 2, 3, 1 / 3]),
    #          FeatureSpec(1, 300, 285, 330, [2, 1 / 2])]
    specs = [FeatureSpec(19, 16, 28, 65, [2, 1 / 2, 3, 1 / 3]),
             FeatureSpec(10, 30, 65, 90, [2, 1 / 2, 3, 1 / 3]),
             FeatureSpec(5, 60, 90, 140, [2, 1 / 2, 3, 1 / 3]),
             FeatureSpec(3, 100, 140, 165, [2, 1 / 2, 3, 1 / 3]),
             FeatureSpec(2, 150, 165, 245, [2, 1 / 2, 3, 1 / 3]),
             FeatureSpec(1, 300, 245, 285, [2, 1 / 2])]
    negatives_per_positive = 3
    min_negatives_per_image = 0

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
        return mobilenet.mobilenet_v2_features(*args, **kwargs)

    @classmethod
    def cls_head_fn(cls, *args, **kwargs):
        return mobilenet.cls_headers(*args, **kwargs)

    @classmethod
    def rgr_head_fn(cls, *args, **kwargs):
        return mobilenet.rgr_headers(*args, **kwargs)


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
    pretrained_weight_path = '/sdb/tmp/pretrained_model/mobilenet_v2_1.0_224.h5'


class VocDenseNetConfig(VocConfig):
    specs = [FeatureSpec(18, 16, 28, 65, [2, 1 / 2, 3, 1 / 3]),
             FeatureSpec(9, 30, 65, 90, [2, 1 / 2, 3, 1 / 3]),
             FeatureSpec(5, 60, 90, 140, [2, 1 / 2, 3, 1 / 3]),
             FeatureSpec(3, 100, 140, 165, [2, 1 / 2, 3, 1 / 3]),
             FeatureSpec(2, 150, 165, 245, [2, 1 / 2, 3, 1 / 3]),
             FeatureSpec(1, 300, 245, 285, [2, 1 / 2])]

    @classmethod
    def feature_fn(cls, *args, **kwargs):
        return densenet.densenet_features(*args, **kwargs)

    @classmethod
    def cls_head_fn(cls, *args, **kwargs):
        return mobilenet.cls_headers(*args, **kwargs)

    @classmethod
    def rgr_head_fn(cls, *args, **kwargs):
        return mobilenet.rgr_headers(*args, **kwargs)

    base_model_name = 'densenet'
    pretrained_weight_path = '/sdb/tmp/pretrained_model/densenet121_weights_tf_dim_ordering_tf_kernels.h5'


class MacConfig(VocConfig):
    voc_path = '/Users/yizuotian/dataset/VOCdevkit/'
    # pretrained_weight_path = '/Users/yizuotian/pretrained_model/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5'
    pretrained_weight_path = 'mobilenet_v2_1.0_224.h5'


cfg = VocDenseNetConfig()
