#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:w_sk time:2019/7/7

from networks import FlowerNet
import keras


num_classes = 5
input_width = 64
input_height = 64
channels_num = 3


def data_process():
    # 为了尽量利用我们有限的训练数据，我们将通过一系列随机变换堆数据进行提升，这样我们的模型将看不到任何两张完全相同的图片，这有利于我们抑制过拟合，使得模型的泛化能力更好
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        # 对图片的每个像素值均乘上这个放缩因子
        rescale=1./255,
        # 错切变换，效果就是让所有点的x坐标(或者y坐标)保持不变，而对应的y坐标(或者x坐标)则按比例发生平移，且平移的大小和该点到x轴(或y轴)的垂直距离成正比
        shear_range=0.2,
        # 让图片在长或宽的方向进行放大，可以理解为某方向的resize.参数大于0小于1时，执行的是放大操作，当参数大于1时，执行的是缩小操作
        zoom_range=0.2,
        # 随机选择图片进行水平翻转
        horizontal_flip=True
    )
    # 以文件夹路径为参数,生成经过数据提升/归一化后的数据,在一个无限循环中无限产生batch数据
    train_generator = train_datagen.flow_from_directory(
        # directory: 目标文件夹路径,对于每一个类,该文件夹都要包含一个子文件夹.子文件夹中任何JPG、PNG、BNP、PPM的图片都会被生成器使用
        './data/train',
        # 所有图像将被resize成该尺寸
        target_size=(input_width, input_height),
        # 每批数据的大小
        batch_size=6,
        # 打乱数据
        shuffle=True,
        # categorical, binary, sparse或None之一. 该参数决定了返回的标签数组的形式
        # 默认为categorical, categorical会返回2D的one-hot编码标签,binary返回1D的二值标签.sparse返回1D的整数标签,如果为None则不返回任何标签, 生成器将仅仅生成batch数据
        class_mode='categorical',
    )

    validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory(
        './data/test',
        target_size=(input_width, input_height),
        batch_size=2,
        class_mode='categorical'
    )

    return train_generator, validation_generator


def train():
    fn = FlowerNet.FlowerNet(input_width, input_height, channels_num)
    flower_model = fn.model()
    train_data, test_data = data_process()
    flower_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0001), metrics=['accuracy'])
    flower_model.fit_generator(generator=train_data, steps_per_epoch=451, validation_data=test_data, validation_steps=480, epochs=20, verbose=1)
    flower_model.save('./models/-model.h5')


train()
