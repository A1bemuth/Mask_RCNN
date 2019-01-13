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

#%matplotlib inline 

# Сюда сохраняем логи и обученную модель
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Путь до файла с весами
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Загружаем веса, если необходимо
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


# ## Настройки

# In[2]:


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
        
        last_id = start_id + count - 1
        
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
        image = Image.open(path)
		image = image.convert('RGB')
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


# In[31]:


#набор для обучения
dataset_train = RocksDataset()
dataset_train.load_images(0, 60)
dataset_train.prepare()

#набор для оценки
dataset_val = RocksDataset()
dataset_val.load_images(60, 3)
dataset_val.prepare()


# ## Создание модели

# In[15]:


# Создаём модель в режиме обучения
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)


# In[41]:


# Веса для начала обучения
init_with = "coco"  # imagenet, coco, или last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Загружаем веса, обученные на базе COCO, но пропускаем
    # слои, отличные по количеству классов.
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # загружаем последнюю обученную модель и продолжаем обучение
    model.load_weights(model.find_last(), by_name=True)


# ## Обучение
# 
# Train in two stages:
# 1. Only the heads. Here we're freezing all the backbone layers and training only the randomly initialized layers (i.e. the ones that we didn't use pre-trained weights from MS COCO). To train only the head layers, pass `layers='heads'` to the `train()` function.
# 
# 2. Fine-tune all layers. For this simple example it's not necessary, but we're including it to show the process. Simply pass `layers="all` to train all layers.

# In[ ]:


# Обучаем головные ветки
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=1, 
            layers='heads')


# In[ ]:


# Тонкая настройка всех слоёв
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=2, 
            layers="all")


# ## Обнаружение

# In[ ]:


class InferenceConfig(RocksConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Пересоздаём модель в режиме обнаружения
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Путь до сохранённых весов
model_path = model.find_last()

# Загружаем обученные веса
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


# In[ ]:


# Тестируем на случайном изображении
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =    modellib.load_image_gt(dataset_val, inference_config, 
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                            dataset_train.class_names, figsize=(8, 8))


# In[ ]:


results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_val.class_names, r['scores'], ax=get_ax())


# ## Оценка

# In[ ]:


# Вычисляем VOC-Style mAP @ IoU=0.5
# Тестируем на 3 изображениях
image_ids = np.random.choice(dataset_val.image_ids, 3)
APs = []
for image_id in image_ids:
    image, image_meta, gt_class_id, gt_bbox, gt_mask =        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Обнаружение объекта
    results = model.detect([image], verbose=0)
    r = results[0]
    # Вычисляем AP
    AP, precisions, recalls, overlaps =        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)
    
print("mAP: ", np.mean(APs))

