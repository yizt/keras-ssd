# -*- coding: utf-8 -*-
"""
 @File    : mobilenet.py
 @Time    : 2019/11/14 上午9:40
 @Author  : yizuotian
 @Description    :
"""
from keras_applications import correct_pad
from tensorflow.python.keras import layers, Model, Input, backend, regularizers

l2_reg = 5e-4


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def mobilenet_v2_base(img_input,
                      alpha=1.0,
                      **kwargs):
    """

    :param img_input:
    :param alpha:
    :param kwargs:
    :return feature1:
    :return feature2:
    """
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    first_block_filters = _make_divisible(32 * alpha, 8)
    x = layers.ZeroPadding2D(padding=correct_pad(backend, img_input, 3),
                             name='Conv1_pad')(img_input)
    x = layers.Conv2D(first_block_filters,
                      kernel_size=3,
                      strides=(2, 2),
                      padding='valid',
                      use_bias=False,
                      kernel_regularizer=regularizers.l2(l2_reg),
                      name='Conv1')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name='bn_Conv1')(x)
    x = layers.ReLU(6., name='Conv1_relu')(x)

    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                            expansion=1, block_id=0)

    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                            expansion=6, block_id=1)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=2)

    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                            expansion=6, block_id=3)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=4)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=5)

    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=2,
                            expansion=6, block_id=6)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=7)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=8)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=9)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=10)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=11)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=12)
    feature1 = x
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=2,
                            expansion=6, block_id=13)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                            expansion=6, block_id=14)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                            expansion=6, block_id=15)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1,
                            expansion=6, block_id=16)
    feature2 = x

    # m = Model(img_input, x)
    # for l in m.layers:
    #     l.trainable = False

    return feature1, feature2


def extra_features(inputs, alpha):
    """
    额外的4个卷积特征用于预测
    :param inputs: [B,H,W,C]
    :param alpha:
    :return feature3:
    :return feature4:
    :return feature5:
    :return feature6:
    """
    feature3 = _inverted_res_block(inputs, expansion=0.2, stride=2,
                                   alpha=alpha, filters=512, block_id='f3')
    feature4 = _inverted_res_block(feature3, expansion=0.25, stride=2,
                                   alpha=alpha, filters=256, block_id='f4')
    feature5 = _inverted_res_block(feature4, expansion=0.5, stride=2,
                                   alpha=alpha, filters=256, block_id='f5')
    feature6 = _inverted_res_block(feature5, expansion=0.25, stride=2,
                                   alpha=alpha, filters=64, block_id='f6')
    return feature3, feature4, feature5, feature6


def mobilenet_v2_features(img_input, alpha=1.):
    f1, f2 = mobilenet_v2_base(img_input, alpha)
    f3, f4, f5, f6 = extra_features(f2, alpha)
    features = [layers.SpatialDropout2D(rate=0.5, name='f{}_dropout'.format(idx + 1))(x)
                for idx, x in enumerate([f1, f2, f3, f4, f5, f6])]
    return features


def cls_headers(feature_list, num_anchors_list, num_classes):
    headers = []
    for i, (feature, num_anchors) in enumerate(zip(feature_list, num_anchors_list)):
        header = seperable_conv2d(feature, num_anchors * num_classes,
                                  'cls_header_{}'.format(i), kernel_size=3)
        # 打平
        header = layers.Reshape(target_shape=(-1, num_classes),
                                name='cls_header_flatten_{}'.format(i))(header)
        headers.append(header)

    # 拼接所有header
    headers = layers.Concatenate(axis=1, name='cls_header_concat')(headers)  # [B,num_anchors,num_classes]
    return headers


def rgr_headers(feature_list, num_anchors_list):
    headers = []
    for i, (feature, num_anchors) in enumerate(zip(feature_list, num_anchors_list)):
        header = seperable_conv2d(feature, num_anchors * 4, 'rgr_header_{}'.format(i), kernel_size=3)
        # 打平
        header = layers.Reshape(target_shape=(-1, 4),
                                name='rgr_header_flatten_{}'.format(i))(header)
        headers.append(header)
    # 拼接所有header
    headers = layers.Concatenate(axis=1, name='rgr_header_concat')(headers)  # [B,num_anchors,4]
    return headers


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
                               depthwise_regularizer=regularizers.l2(l2_reg),
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
                      kernel_regularizer=regularizers.l2(l2_reg),
                      name=prefix + 'project')(x)
    return x


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    in_channels = backend.int_shape(inputs)[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if block_id:
        # Expand
        x = layers.Conv2D(round(expansion * in_channels),
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          activation=None,
                          kernel_regularizer=regularizers.l2(l2_reg),
                          name=prefix + 'expand')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      epsilon=1e-3,
                                      momentum=0.999,
                                      name=prefix + 'expand_BN')(x)
        x = layers.ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    if stride == 2:
        x = layers.ZeroPadding2D(padding=correct_pad(backend, x, 3),
                                 name=prefix + 'pad')(x)
    x = layers.DepthwiseConv2D(kernel_size=3,
                               strides=stride,
                               activation=None,
                               use_bias=False,
                               depthwise_regularizer=regularizers.l2(l2_reg),
                               padding='same' if stride == 1 else 'valid',
                               name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'depthwise_BN')(x)

    x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = layers.Conv2D(pointwise_filters,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      activation=None,
                      kernel_regularizer=regularizers.l2(l2_reg),
                      name=prefix + 'project')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'project_BN')(x)

    if in_channels == pointwise_filters and stride == 1:
        return layers.Add(name=prefix + 'add')([inputs, x])
    return x


if __name__ == '__main__':
    # m = keras.applications.MobileNetV2(input_shape=(300, 300, 3),
    #                                    weights=None)
    # m.summary()

    # inputs = layers.Input(shape=(10,))
    # a = layers.Dense(units=20, name='a')(inputs)
    # b = layers.Dense(units=40, name='b')(a)
    # c = layers.Dense(units=60, name='c')(a)
    # m = Model(inputs, c)
    # m.summary()
    im_input = Input(shape=(300, 300, 3))
    fs = mobilenet_v2_features(im_input, alpha=1.)
    m = Model(im_input, fs)
    m.summary()
