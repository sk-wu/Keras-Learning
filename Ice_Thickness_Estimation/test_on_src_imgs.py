# 在原图上检验模型的定位效果

import os
import cv2
import keras
import numpy as np

model_path = './models/loc_model/loc_model_20.h5'
test_img_path = './data/test/test2'
width = 240
height = 27
channels_num = 3


def preprocess(src_img):
    norm_img = src_img / 255
    resize_img = cv2.resize(norm_img, (width, height))
    reshape_img = np.resize(resize_img, (height, width, channels_num))
    return reshape_img


def show_single_img():
    my_model = keras.models.load_model(model_path)
    my_model.summary()
    print("Successfully Load Model!")
    for file in os.listdir(test_img_path):
        img = cv2.imread(os.path.join(test_img_path, file))
        print(file)
        h, w = img.shape[:2]
        cut_img = img[int(0.8 * h):, :]
        cut_img = preprocess(cut_img)
        input = np.expand_dims(cut_img, 0)
        result = my_model.predict(input)
        print(result)
        cv2.line(img, (0, int(0.8*h)), (w, int(0.8*h)), (255, 0, 0), 2)
        cv2.line(img, (0, int(h)-1), (w, int(h)-1), (255, 0, 0), 2)
        cv2.line(img, (int(result[0][0] * w), int(0.8 * h)), (int(result[0][3] * w), h), (255, 0, 0), 2)
        cv2.line(img, (int(result[0][1] * w), int(0.8 * h)), (int(result[0][2] * w), h), (255, 0, 0), 2)
        cv2.namedWindow("ttt", cv2.WINDOW_NORMAL)
        cv2.imshow("ttt", img)
        cv2.waitKey()


# show_single_img()
def save_img_results():
    my_model = keras.models.load_model(model_path)
    my_model.summary()
    print("Successfully Load Model!")
    for file in os.listdir(test_img_path):
        img = cv2.imread(os.path.join(test_img_path, file))
        print(file)
        img = cv2.resize(img, (1920, 1080))
        h, w = img.shape[:2]
        cut_img = img[int(0.8 * h):, :]
        cut_img = preprocess(cut_img)
        input = np.expand_dims(cut_img, 0)
        result = my_model.predict(input)
        print(result)
        cv2.line(img, (0, int(0.8*h)), (w, int(0.8*h)), (255, 0, 0), 2)
        cv2.line(img, (0, int(h)-1), (w, int(h)-1), (255, 0, 0), 2)
        cv2.line(img, (int(result[0][0] * 8), int(0.8 * h)), (int(result[0][3] * 8), h), (255, 0, 0), 2)
        cv2.line(img, (int(result[0][1] * 8), int(0.8 * h)), (int(result[0][2] * 8), h), (255, 0, 0), 2)
        cv2.imwrite(os.path.join("./results_on_src_imgs", file), img)

save_img_results()