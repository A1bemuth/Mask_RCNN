#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import pickle
from PIL import Image

# Корневая папка проекта
ROOT_DIR = os.path.abspath("../../")

# Путь папки с изображениями
IMAGES_DIR = "./images"

# Загружаем библиотеки алгоритма Mask RCNN
sys.path.append(ROOT_DIR)  # Чтоб находить файлы библиотеки
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# Сюда сохраняем логи и обученную модель
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


class RocksConfig(Config):
    """Конфигурация для обучения на датасете камней
    """
    NAME = "rocks"

    # При запуске на CPU нужно указывать GPU_COUNT = 1. Изображения
    # не самые маленькие, поэтому всего 2 изображения за раз. 
    # 12GB GPU может обрабатывать 2 изображения 1024x1024px.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Количество классов (включая задний фон)
    NUM_CLASSES = 1 + 1  # фон + камень

    # Изображения увеличиваются по меньшей стороне до IMAGE_MIN_DIM
    # так, чтоб большая сторона не превысила IMAGE_MAX_DIM. Затем
    # добиваются нулями до квадрата.
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 448
    
config = RocksConfig()
# config.display()


# ## Датасет

# In[21]:


#Этот класс загружает изображения и маски датасета
#Для использования переопределяются load_image,
#load_mask, image_reference
class RocksDataset(utils.Dataset):
    
    #перечисляет изображения в images/ и сохраняет информацию о них
    def load_images(self, start_id, count):
        # Add classes
        self.add_class("rocks", 1, "rock")
        
        last_id = start_id + count - 1;
        
        model_id = 0
        for file in os.listdir(IMAGES_DIR):
            match = re.match("^(\d+)\.png$", file)
            if match:
                image_id = int(match.group(1))
                if image_id >= start_id and image_id <= last_id:
                    self.add_image("rocks", image_id=model_id,
                                   real_image_id=int(match.group(1)), 
                                   path=f"{IMAGES_DIR}/{file}")
                    print(f"Add image {image_id} with id {model_id}")
                    model_id += 1

    #загружает изображение в виде массива numpy
    def load_image(self, image_id):
        print(f"Asked to find image at {image_id}")
        path = self.image_reference(image_id)["path"]
        return self.load_png_as_nparray(path)
    
    #загружает маску в виде кортежа. первый элемент - массив boolean
    #формы width/height/masks_num, второй - массив с номерами классов каждой маски
    #размера masks_num
    def load_mask(self, index):
        print(f"Asked to find mask at {index}")
        index = self.image_reference(index)["real_image_id"]
        mask = self.get_mask(index)
        if mask:
            return mask;
        masks = []
        class_ids = []
        for file in os.listdir(IMAGES_DIR):
            match = re.match(f"^{index}_mask_(\d+)_(\d+)\.png$", file)
            if match:
                mask_img, w, h = self.load_png(f"{IMAGES_DIR}/{file}")
                bool_mask = self.pixels_to_mask(mask_img)
                bool_mask = bool_mask.reshape(h, w)
                masks.append(bool_mask)
                class_ids.append(int(match.group(1)))
        merged_mask = np.stack(masks, axis=2)
        masks_num = merged_mask.shape[2]
        mask = (merged_mask, np.array(class_ids))
        self.save_mask(index, mask)
        return mask
    
    #кэш масок. загружает маску из памяти, или с диска
    #если не нашли - собираем заново из изображений
    #в images/
    def get_mask(self, index):
        try:
            try:
                return self._masks[index]
            except AttributeError:
                self._masks = {};
            except KeyError:
                print("loading mask from disk")
            serialized = open(f"{MODEL_DIR}/{index}.mask", "rb").read()
            mask = pickle.loads(serialized)
            self._masks[index] = mask
            return mask
        except:
            return None
    
    #сериализует маску на диск, чтоб не собирать при новом прогоне
    def save_mask(self, index, mask):
        serialized = pickle.dumps(mask)
        open(f"{MODEL_DIR}/{index}.mask", "wb").write(serialized)
        try:
            self._masks[index] = mask
        except AttributeError:
            self._masks = {};

    #возвращает информацию об изображении
    def image_reference(self, image_id):
        #print(f"Asked for an image reference at {image_id}")
        """Return the shapes data of the image."""
        for info in self.image_info:
            if info["id"] == image_id:
                return info;
        super(self.__class__, self).image_reference(image_id)

    def load_png_as_nparray(self, path):
#         print(path)
        image = Image.open(path)
        w, h = image.size
        return np.array(image.getdata()).reshape(h, w, 3).astype(np.uint8)
    
    def load_png(self, path):
        print(path)
        image = Image.open(path)
        w, h = image.size
        return (np.array(image.getdata()).astype(np.uint8), w, h)
        
    def image_to_mask(self, img):
        x, y, z = img.shape
        reshaped = img.reshape(x * y, z)
        to_bool = lambda p: False if any(v > 0 for v in p) else True
        img = np.array([to_bool(pixel) for pixel in reshaped])
        return img.reshape(x, y)
    
    def pixels_to_mask(self, img):
        to_bool = lambda p: False if any(v > 0 for v in p) else True
        return np.array([to_bool(pixel) for pixel in img])