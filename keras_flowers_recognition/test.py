#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:w_sk time:2019/7/7

import keras
import cv2
import os
import numpy as np

img_dir = './img'


def test():
    flowers = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    my_model = keras.models.load_model('./models/-model.h5')
    my_model.summary()

    for file in os.listdir(img_dir):
        src = cv2.imread(os.path.join(img_dir, file))
        img = cv2.resize(src, (64, 64))
        # 训练时输入数据的格式为(batch_size, input_width, input_height, channels_num),因此需要增加一个元素
        img = np.expand_dims(img, 0)
        result = my_model.predict(img)
        index = np.argmax(result)
        print(flowers[index])
        cv2.putText(src, flowers[index], (50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), 2, 5)
        cv2.imshow("img", src)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


test()
