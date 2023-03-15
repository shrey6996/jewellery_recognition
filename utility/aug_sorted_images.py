import os
import shutil

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img

class AugSortedImages():
    def __init__(self,
                 db_path = "data/sorted_db2",
                 output_path = "data",
                 op_folder_name = "aug-crop-db",
                 jewellery_types = {"RG" : "Ring", 
                                    "NK" : "Necklace", 
                                    "BR" : "Bracelate", 
                                    "ER" : "Earring",
                                    "OT" : "Other"},
                 ):
        self.db_path = db_path
        self.output_path = output_path
        self.op_folder_name = op_folder_name
        self.jewellery_types = jewellery_types

    def aug(self):
        datagen = ImageDataGenerator(
                featurewise_center=True, 
                featurewise_std_normalization=True, 
                zca_whitening=True, 
                zca_epsilon=1e-8,
                rotation_range=30,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.1,
                horizontal_flip=True,
                vertical_flip=True,
                brightness_range=(0.4, 0.6),
                channel_shift_range=30.0,
                fill_mode='nearest')

        all_images = os.listdir(self.db_path)

        #make folder with subfolder of all jewellery type
        dirpath = os.path.join(self.output_path, self.op_folder_name)
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)

        os.mkdir(os.path.join(self.output_path, self.op_folder_name))

        for i in self.jewellery_types.values():
            os.mkdir(os.path.join(self.output_path, self.op_folder_name, i))

        for j in self.jewellery_types.values():
            for files in os.listdir(os.path.join(self.db_path, j)):
                try:
                    img = load_img(os.path.join(self.db_path, j, files))  # this is a PIL image
                except Exception as e:
                    print(e)
                    continue

                x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
                x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

                # the .flow() command below generates batches of randomly transformed images
                # and saves the results to the `preview/` directory
                i = 0
                for batch in datagen.flow(x, batch_size=1,
                                            save_to_dir=os.path.join(self.output_path, self.op_folder_name, j), save_prefix=f'{files}___', save_format='jpeg'):
                    i += 1
                    if i > 15:
                        break  # otherwise the generator would loop indefinitely
                print(files)

        print("Done!")
