# 剪切图片
import cv2
import os

for file in os.listdir("./data_imgs/src_imgs"):
    if file.endswith(".jpg"):
        img = cv2.imread(os.path.join("./data_imgs/src_imgs", file))
        r_img = img[int(0.8*img.shape[0]):, :]
        new_path = "./data_imgs/cut_imgs"
        cv2.imwrite(os.path.join(new_path, file), r_img)