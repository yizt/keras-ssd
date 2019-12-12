# -*- coding: utf-8 -*-
"""
 @File    : densenet.py
 @Time    : 2019/11/29 下午6:43
 @Author  : yizuotian
 @Description    :
"""
from tensorflow.python.keras import layers, backend, regularizers

l2_reg = 5e-4


def densenet_base(img_input,
                  blocks=[6, 12, 24, 16],
                  **kwargs):
    bn_axis = 3

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False,
                      kernel_regularizer=regularizers.l2(l2_reg),
                      name='conv1/conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
    x = layers.Activation('relu', name='conv1/relu')(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    _, x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    _, x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    f1, x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')

    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
    x = layers.Activation('relu', name='relu')(x)
    f2 = x
    return f1, f2


def extra_features(inputs):
    """
    额外的4个卷积特征用于预测
    :param inputs: [B,H,W,C]
    :return feature3:
    :return feature4:
    :return feature5:
    :return feature6:
    """
    x = conv_unit(inputs, 256, 1, stride=1, name='f3_1')
    x = conv_unit(x, 256, 3, stride=2, name='f3_2')
    feature3 = x
    x = conv_unit(x, 128, 1, stride=1, name='f4_1')
    x = conv_unit(x, 256, 3, stride=2, name='f4_2')
    feature4 = x
    x = conv_unit(x, 128, 1, stride=1, name='f5_1')
    x = conv_unit(x, 256, 3, stride=2, name='f5_2')
    feature5 = x
    x = conv_unit(x, 128, 1, stride=1, name='f6_1')
    x = conv_unit(x, 256, 3, stride=2, name='f6_2')
    feature6 = x

    return feature3, feature4, feature5, feature6


def dense_block(x, blocks, name):
    """A dense block.

    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):
    """A transition block.

    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    bn_axis = 3
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      kernel_regularizer=regularizers.l2(l2_reg),
                      name=name + '_conv')(x)
    conv_before_pool = x
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return conv_before_pool, x


def conv_block(x, growth_rate, name):
    """A building block for a dense block.

    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.

    # Returns
        Output tensor for the block.
    """
    bn_axis = 3
    x1 = layers.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = layers.Conv2D(4 * growth_rate, 1,
                       use_bias=False,
                       kernel_regularizer=regularizers.l2(l2_reg),
                       name=name + '_1_conv')(x1)
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       kernel_regularizer=regularizers.l2(l2_reg),
                       name=name + '_2_conv')(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def conv_unit(x, filters, kernel_size, stride=1, name=None):
    bn_axis = 3
    x1 = layers.Conv2D(filters,
                       kernel_size=kernel_size,
                       padding='same',
                       use_bias=False,
                       strides=stride,
                       kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(l2_reg),
                       name=name + 'conv')(x)
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_bn')(x1)
    x1 = layers.Activation('relu', name=name + '_relu')(x1)
    return x1


def densenet_features(img_input):
    f1, f2 = densenet_base(img_input)
    f3, f4, f5, f6 = extra_features(f2)
    # m = Model(img_input, f2)
    # for l in m.layers:
    #     l.trainable = False
    features = [layers.SpatialDropout2D(rate=0.8, name='f{}_dropout'.format(idx + 1))(x)
                for idx, x in enumerate([f1, f2, f3, f4, f5, f6])]

    return features


def cls_headers(feature_list, num_anchors_list, num_classes):
    headers = []
    for i, (feature, num_anchors) in enumerate(zip(feature_list, num_anchors_list)):
        header = layers.Conv2D(num_anchors * num_classes,
                               kernel_size=3,
                               padding='same',
                               use_bias=False,
                               kernel_initializer='he_normal',
                               kernel_regularizer=regularizers.l2(l2_reg),
                               name='cls_header_{}'.format(i))(feature)
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
        header = layers.Conv2D(num_anchors * 4,
                               kernel_size=3,
                               padding='same',
                               use_bias=False,
                               kernel_initializer='he_normal',
                               kernel_regularizer=regularizers.l2(l2_reg),
                               name='rgr_header_{}'.format(i))(feature)
        # 打平
        header = layers.Reshape(target_shape=(-1, 4),
                                name='rgr_header_flatten_{}'.format(i))(header)
        headers.append(header)
    # 拼接所有header
    headers = layers.Concatenate(axis=1, name='rgr_header_concat')(headers)  # [B,num_anchors,4]
    return headers


if __name__ == '__main__':
    from tensorflow.python.keras import Input, Model

    im_input = Input(shape=(300, 300, 3))
    o = densenet_features(im_input)
    m = Model(im_input, list(o))
    m.summary()
