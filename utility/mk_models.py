import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import pandas as pd

import json
import os
import pickle
import shutil
import time

# model_url = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"
model_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"
IMAGE_SHAPE = (224, 224)
layer = hub.KerasLayer(model_url)
model = tf.keras.Sequential([layer])

class MkModel():
    def __init__(self,
                 model_name = "real.pkl",
                 sorted_database_image_path = "real",
                 jewellery_types = {
                    "RG" : "Ring", 
                    "NK" : "Necklace", 
                    "BR" : "Bracelate", 
                    "ER" : "Earring",
                    "OT" : "Other"
                 },
            ):
        self.model_path = os.path.join("model", model_name)
        self.database_image_path = sorted_database_image_path
        self.jewellery_types = jewellery_types

    def feature_extractor(self, img_path):
        file = Image.open(img_path).convert('L').resize(IMAGE_SHAPE)
        # file = ImageOps.invert(file)
        # enhancer = ImageEnhance.Sharpness(file)
        # factor = 3
        # file = enhancer.enhance(factor)
        file = np.stack((file,)*3, axis=-1)
        file = np.array(file)/255.0

        embedding = model.predict(file[np.newaxis, ...])

        vgg16_feature_np = np.array(embedding)
        flattended_feature = vgg16_feature_np.flatten()

        return flattended_feature

    def mk_model(self):
        start = time.time()
        all_results = dict()

        for i in self.jewellery_types.values():
            all_results[i] = dict()

        for i in self.jewellery_types.values():
            for count, image in enumerate(os.listdir(os.path.join(self.database_image_path, i))):
                try:
                    result = self.feature_extractor(os.path.join(self.database_image_path, i, image))
                    all_results[i][image] = result

                except Exception as e:
                    print(e)

        # print(all_results)

        with open(self.model_path,'wb') as file:
            pickle.dump(all_results, file)

        end = time.time()

        print("total time:", end-start)
