'''
旋转角度。随机旋转图像一定角度; 改变图像内容的朝向。
随机颜色。对图像进行颜色抖动，对图像的每个像素RGB进行随机扰动, 常用的噪声模式是椒盐噪声和高斯噪声。
对比度增强。增强图像对比度，也可以用直方图均衡化。
亮度增强。将整个图像亮度调高。
颜色增强。
椒盐噪声。
高斯噪声。
参考：1、https://blog.csdn.net/comway_Li/article/details/82928974
2、https://www.cnblogs.com/lfri/p/10627595.html
'''

from PIL import Image
from PIL import ImageEnhance
import os
import numpy as np


# 旋转角度
def rotation(root_path, img_name):
    img = Image.open(os.path.join(root_path, img_name))
    rotation_img = img.rotate(20)
    # rotation_img.save(os.path.join(root_path,img_name.split('.')[0] + '_rotation.jpg'))
    return rotation_img


# 随机颜色,对图像进行颜色抖动
def randomColor(root_path, img_name):
    image = Image.open(os.path.join(root_path, img_name))
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度


# 对比度增强
def contrastEnhancement(root_path, img_name):
    image = Image.open(os.path.join(root_path, img_name))
    enh_con = ImageEnhance.Contrast(image)
    contrast = 1.5
    image_contrasted = enh_con.enhance(contrast)
    return image_contrasted


# 亮度增强
def brightnessEnhancement(root_path, img_name):  # 亮度增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 1.5
    image_brightened = enh_bri.enhance(brightness)


# 颜色增强
def colorEnhancement(root_path, img_name):
    image = Image.open(os.path.join(root_path, img_name))
    enh_col = ImageEnhance.Color(image)
    color = 1.5
    image_colored = enh_col.enhance(color)
    return image_colored


# 添加椒盐噪声，num为个数
def sp_noise(root_path, img_name, num=2500):
    image = Image.open(os.path.join(root_path, img_name))
    img = np.array(image)
    # 随机生成num个椒盐点
    rows, cols, dims = img.shape
    for i in range(num):
        x = np.random.randint(0, rows)
        y = np.random.randint(0, cols)
        img[x, y, :] = 0
    for i in range(num):
        x = np.random.randint(0, rows)
        y = np.random.randint(0, cols)
        img[x, y, :] = 255
    img = Image.fromarray(img)
    return img


# 添加高斯噪声,参数为均值和方差
def gaussian_noise(root_path, img_name, mean=0, var=0.05):
    image = Image.open(os.path.join(root_path, img_name))
    image = np.array(image)
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    out = Image.fromarray(out)
    return out


imageDir = "./resize_img"  # 原始图片的路径文件夹
saveDir = "./argumentation/new"  # 要保存的图片的路径文件夹
count = 25000
for name in os.listdir(imageDir):
    count = count + 1
    saveName = str(count) + ".jpg"
    # saveImage = gaussian_noise(imageDir, name)
    saveImage = sp_noise(imageDir, name, 100000)
    saveImage.save(os.path.join(saveDir, saveName))




