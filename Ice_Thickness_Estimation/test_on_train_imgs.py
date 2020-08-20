# 在训练图片上检验模型的定位效果

import keras
import numpy as np
import cv2
import os

input_width = 240
input_height = 27
channels_num = 3


model_path = './models/loc_model/loc_model_15.h5'
img_path = './data/location/images'


def preprocess(src_img):
    norm_img = src_img / 255
    resize_img = cv2.resize(norm_img, (input_width, input_height))
    reshape_img = np.resize(resize_img, (input_height, input_width, channels_num))
    return reshape_img


if __name__ == '__main__':
    my_model = keras.models.load_model(model_path)
    my_model.summary()
    for index, element in enumerate(os.listdir(img_path)):
        src_img = cv2.imread(os.path.join(img_path, element))
        img = cv2.resize(src_img, (input_width, input_height))
        h, w = img.shape[:2]
        new_img = preprocess(img)
        input = np.expand_dims(new_img, 0)
        result = my_model.predict(input)
        print(result)
        cv2.line(src_img, (int(result[0][0] * 8), 0), (int(result[0][3] * 8), 216), (0, 0, 255), 6)
        cv2.line(src_img, (int(result[0][1] * 8), 0), (int(result[0][2] * 8), 216), (0, 0, 255), 6)
        cv2.imwrite(os.path.join("./results_on_train_imgs", element), src_img)