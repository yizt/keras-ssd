# -*- coding: utf-8 -*-
"""
 @File    : train.py
 @Time    : 2019/11/24 下午6:57
 @Author  : yizuotian
 @Description    :
"""

import argparse
import sys

import tensorflow as tf
import time
from tensorflow.python.keras import backend
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, \
    EarlyStopping

from config import cfg
from datasets.dataset import VocDataset
from ssd import ssd_model
from utils import model_utils
from utils.generator import Generator
from utils.preprocess import TrainAugmentation, EvalTransform


def set_gpu_growth():
    config = tf.ConfigProto(allow_soft_placement=True)  # because no supported kernel for GPU devices is available
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    backend.set_session(session)


def lr_schedule(total_epoch, lr):
    def _lr_fn(epoch):
        if epoch < total_epoch * 0.6:
            return lr
        elif epoch < total_epoch * 0.8:
            return lr / 10.
        else:
            return lr / 100

    return _lr_fn


def get_call_back(lr, batch_size):
    """
    定义call back
    :return:
    """
    text = '{}-{}-{}'.format(cfg.base_model_name, batch_size, lr)
    checkpoint = ModelCheckpoint(filepath='/tmp/ssd-' + text + '.{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 save_freq='epoch'
                                 )
    reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.2, min_delta=2e-3,
                               patience=10, min_lr=1e-5)
    stop = EarlyStopping(monitor='val_loss', patience=20)
    # scheduler = LearningRateScheduler(lr_schedule(epochs, lr))

    log = TensorBoard(log_dir='log-{}-{}'.format(text,
                                                 time.strftime("%Y%m%d", time.localtime())))
    return [checkpoint, reduce, stop, log]


def main(args):
    set_gpu_growth()
    dataset = VocDataset(cfg.voc_path, class_mapping=cfg.class_mapping)
    dataset.prepare()
    train_img_info = [info for info in dataset.get_image_info_list() if info.type == 'trainval']  # 训练集
    print("train_img_info:{}".format(len(train_img_info)))
    test_img_info = [info for info in dataset.get_image_info_list() if info.type == 'test']  # 测试集
    print("test_img_info:{}".format(len(test_img_info)))

    m = ssd_model(cfg.feature_fn, cfg.cls_head_fn, cfg.rgr_head_fn, cfg.input_shape,
                  cfg.num_classes, cfg.specs, cfg.max_gt_num,
                  cfg.positive_iou_threshold, cfg.negative_iou_threshold,
                  cfg.negatives_per_positive, cfg.min_negatives_per_image)

    # 加载预训练模型
    init_epoch = args.init_epoch
    if args.init_epoch > 0:
        text = '{}-{}-{}'.format(cfg.base_model_name, args.batch_size, args.lr)
        m.load_weights('/tmp/ssd-{}.{:03d}.h5'.format(text, init_epoch), by_name=True)
    else:
        m.load_weights(cfg.pretrained_weight_path, by_name=True)
    # 生成器
    transforms = TrainAugmentation(cfg.image_size, cfg.mean_pixel, cfg.std)
    train_gen = Generator(train_img_info,
                          transforms,
                          cfg.input_shape,
                          args.batch_size,
                          cfg.max_gt_num)
    # 生成器
    val_trans = TrainAugmentation(cfg.image_size, cfg.mean_pixel, cfg.std)
    val_gen = Generator(test_img_info,
                        val_trans,
                        cfg.input_shape,
                        args.batch_size,
                        cfg.max_gt_num)
    model_utils.compile(m, args.lr, args.momentum, args.clipnorm, args.weight_decay,
                        cfg.loss_weights)
    m.summary()

    # 训练
    m.fit_generator(train_gen,
                    epochs=args.epochs,
                    verbose=1,
                    initial_epoch=init_epoch,
                    validation_data=val_gen,
                    use_multiprocessing=False,
                    workers=10,
                    callbacks=get_call_back(args.lr, args.batch_size))


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--batch-size", type=int, default=16, help="batch size")
    parse.add_argument("--epochs", type=int, default=800, help="epochs")
    parse.add_argument("--init-epoch", type=int, default=0, help="init epoch")
    parse.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parse.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parse.add_argument("--weight-decay", type=float, default=5e-4, help="weight decay")
    parse.add_argument("--clipnorm", type=float, default=1.,
                       help="Gradients will be clipped when their L2 norm exceeds this value")
    arguments = parse.parse_args(sys.argv[1:])
    main(arguments)
