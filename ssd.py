# -*- coding: utf-8 -*-
"""
 @File    : ssd.py
 @Time    : 2019/11/14 下午4:30
 @Author  : yizuotian
 @Description    :
"""
from tensorflow.python.keras import backend, layers, Model


def seperable_conv2d(x, filters, name, kernel_size=1, stride=1, padding='same'):
    """

    :param x:
    :param filters:
    :param name:
    :param kernel_size:
    :param stride:
    :param padding:
    :return:
    """
    prefix = name
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    x = layers.DepthwiseConv2D(kernel_size=kernel_size,
                               strides=stride,
                               activation=None,
                               use_bias=False,
                               padding=padding,
                               name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'depthwise_BN')(x)

    x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = layers.Conv2D(filters,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      activation=None,
                      name=prefix + 'project')(x)
    return x


def cls_headers(feature_list, num_classes):
    headers = []
    for i, feature in enumerate(feature_list):
        header = seperable_conv2d(feature, 6 * num_classes,
                                  'cls_header_{}'.format(i), kernel_size=3)
        # 打平
        header = layers.Reshape(target_shape=(-1, 6 * num_classes),
                                name='cls_header_flatten_{}'.format(i))(header)
        headers.append(header)

    # 拼接所有header
    headers = layers.Concatenate(axis=0, name='cls_header_concat')(headers)
    return headers


def rgr_headers(feature_list):
    headers = []
    for i, feature in enumerate(feature_list):
        header = seperable_conv2d(feature, 6 * 4, 'rgr_header_{}'.format(i), kernel_size=3)
        # 打平
        header = layers.Reshape(target_shape=(-1, 6 * 4),
                                name='rgr_header_flatten_{}'.format(i))(header)
        headers.append(header)
    # 拼接所有header
    headers = layers.Concatenate(axis=0, name='rgr_header_concat')(headers)
    return headers


def ssd_model(input_shape, feature_fn, num_classes, stage='train'):
    image_input = layers.Input(shape=input_shape)
    feature_list = feature_fn(image_input)

    class_logits = cls_headers(feature_list, num_classes)
    deltas = rgr_headers(feature_list)

    m = Model(image_input, [class_logits, deltas])
    return m


if __name__ == '__main__':
    from base_net.mobilenet import mobilenet_v2_features

    model = ssd_model((300, 300, 3), mobilenet_v2_features, 3)
    model.summary()
