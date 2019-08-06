#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:w_sk time:2019/4/3

import cv2
import numpy as np
import os
import time
from keras.models import load_model

test_img_dir = "./images"


def predict_on_single_image():
    animal_dict = ['cat', 'dog']
    my_model = load_model('./models/dc_model.h5')
    for file in os.listdir(test_img_dir):
        time_start = time.time()
        src_image = cv2.imread(os.path.join(test_img_dir, file))
        test_image = cv2.resize(src_image, (128, 128))
        img = np.expand_dims(test_image, 0)
        result = my_model.predict(img)
        index = np.argmax(result)
        print(animal_dict[index])
        time_end = time.time()
        print('time:', time_end - time_start)
        cv2.imshow("img", src_image)
        cv2.waitKey()


predict_on_single_image()

