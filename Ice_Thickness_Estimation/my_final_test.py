

import os
import cv2
import keras
import numpy as np
import time

loc_model_path = './models/loc_model/loc_model_20.h5'  # 检测模型
classify_model_path = './models/classify_model/classify_model_10.h5'  # 分类模型
test_img_path = './data/test/test2'  # 测试图片地址
output_path = "./results_on_src_imgs_0"  # 测试图片输出地址
width = 240
height = 27
channels_num = 3


def preprocess(src_img):
    norm_img = src_img / 255
    resize_img = cv2.resize(norm_img, (width, height))
    reshape_img = np.resize(resize_img, (height, width, channels_num))
    return reshape_img


def test():
    print("Load Classification Model...")
    classify_model = keras.models.load_model(classify_model_path)
    classify_model.summary()
    print("Load Location Model...")
    loc_model = keras.models.load_model(loc_model_path)
    loc_model.summary()
    for file in os.listdir(test_img_path):
        start = time.time()
        img = cv2.imread(os.path.join(test_img_path, file))
        img = cv2.resize(img, (1920, 1080))
        print(file)
        h, w = img.shape[:2]
        cut_img = img[int(0.8 * h):, :]
        cut_img = preprocess(cut_img)
        input = np.expand_dims(cut_img, 0)
        classify_result = classify_model.predict(input)
        index = np.argmax(classify_result)
        print(index)
        if not index:

            result = loc_model.predict(input)
            print(result)
            cv2.line(img, (0, int(0.8*h)), (w, int(0.8*h)), (255, 0, 0), 6)
            cv2.line(img, (0, int(h)-1), (w, int(h)-1), (255, 0, 0), 6)
            cv2.line(img, (int(result[0][0] * 8), int(0.8 * h)), (int(result[0][3] * 8), h), (0, 0, 255), 6)
            cv2.line(img, (int(result[0][1] * 8), int(0.8 * h)), (int(result[0][2] * 8), h), (0, 0, 255), 6)
            print(time.time()-start)

        else:
            text = "Warning!"
            textSize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 4, 2)
            x = w//2 - textSize[0][0]//2
            y = h//2 - textSize[0][1]//2
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(output_path, file), img)

test()