#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:w_sk time:2019/4/2

import keras
from networks import RecNet

num_classes = 2
train_data_dir = './dataset/train'
validation_data_dir = './dataset/validation'
input_width = 128
input_height = 128
channels_num = 3


def train_model():
    train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                 rotation_range=20,
                                                                 width_shift_range=0.2,
                                                                 height_shift_range=0.2,
                                                                 shear_range=0.2,
                                                                 zoom_range=0.5,
                                                                 horizontal_flip=True,
                                                                 fill_mode='nearest')

    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        target_size=(128, 128),
                                                        batch_size=32,
                                                        shuffle=True,
                                                        class_mode='categorical')

    validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                                  target_size=(128, 128),
                                                                  batch_size=24,
                                                                  class_mode='categorical')

    my_earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, mode='min')
    my_checkpoint = keras.callbacks.ModelCheckpoint('./model/dc_model.h5', monitor='val_acc', verbose=0, save_best_only=True, mode='max', period=1)
    my_tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', write_graph=True)
    my_callbacks = [my_earlystopping, my_checkpoint, my_tensorboard]

    dc_model = RecNet.RecNet(input_width, input_height, channels_num).model()
    history = dc_model.fit_generator(train_generator, epochs=100, steps_per_epoch=80, validation_data=validation_generator, validation_steps=42, callbacks=my_callbacks)


train_model()








