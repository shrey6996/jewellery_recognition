import os
import shutil

import torch
from PIL import Image

class CropDB():
    def __init__(self, db_path, path, op_folder):
        self.db_path = db_path
        self.path = path
        self.op_folder = op_folder
        
        device = torch.device("cpu")
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path="model/exp29.pt", force_reload=True, device=device)
        
    def crop(self):
        if os.path.exists(os.path.join(self.path, self.op_folder)) and os.path.isdir(os.path.join(self.path, self.op_folder)):
            shutil.rmtree(os.path.join(self.path, self.op_folder))
            
        os.mkdir(os.path.join(self.path, self.op_folder))
                     
        all_img = []
        for img in os.listdir(self.db_path):
            try:
                image = Image.open(os.path.join(self.db_path, img))
            except:
                print(img, "not open")
                continue
                
            try:
                results = self.yolo_model([image])
                f = results.pandas().xyxy[0].sort_values(by=['confidence'], ascending=False).iloc[0]

                img2 = image.crop(
                    (f['xmin'] + (f['xmin'] * 0.30), 
                     f['ymin'] + (f['ymin'] * 0.30), 
                     f['xmax'] + (f['xmax'] * 0.30), 
                     f['ymax'] + (f['ymax'] * 0.30)))
                
                width, height = img2.size
                
                if(width != height):
                    bigside = width if width > height else height

                    background = Image.new('RGB', (bigside, bigside), (0, 0, 0))
                    offset = (int(round(((bigside - width) / 2), 0)), int(round(((bigside - height) / 2),0)))

                    background.paste(img2, offset)
                    background.save(os.path.join(self.path, self.op_folder, img))

                    print(img, "Image has been resized !")

                else:
                    print(img, "Image is already a square, it has not been resized !")

                    img2.save(fp=os.path.join(self.path, self.op_folder, img))
                    
                
            except Exception as e:
                print(img, e)
                image.save(fp=os.path.join(self.path, self.op_folder, img))

            
            
#             try:
#                 all_img.append(Image.open(os.path.join(self.db_path, img)))
#             except:
#                 print(img)
        
#         results = self.yolo_model(all_img)
#         results.crop(save=True, save_dir=os.path.join(self.path, "temp"))
        
#         shutil.move(os.path.join(self.path, "temp/crops/j"), os.path.join(self.path, self.op_folder))
#         shutil.rmtree(os.path.join(self.path, "temp"))