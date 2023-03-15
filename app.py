import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from scipy.spatial import distance

import os
import pickle
import shutil
import io
from io import BytesIO  

from flask import Flask, render_template, request
import torch

app = Flask(__name__)

app.config['SECRET_KEY'] = os.urandom(24)
saved_imgs = []

model_name = "client_samyak.pkl"
model_path = os.path.join("model", model_name)

# give sorted db path (not augmented)
# database_images_path = "data/sorted_combined"
database_images_path = "data/client_samyak_sorted"

metric = 'cosine'

# model_url = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"
model_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"
IMAGE_SHAPE = (224, 224)

layer = hub.KerasLayer(model_url)
model = tf.keras.Sequential([layer])

def feature_extractor(img_path):
    file = Image.open(img_path).convert('L').resize(IMAGE_SHAPE)
    #file = ImageOps.invert(file)
    #enhancer = ImageEnhance.Sharpness(file)
    #factor = 3
    #file = enhancer.enhance(factor)
    file = np.stack((file,)*3, axis=-1)
    file = np.array(file)/255.0

    with tf.device('/cpu:0'):
        embedding = model.predict(file[np.newaxis, ...])
    flattended_feature = np.array(embedding).flatten()

    return flattended_feature
    


def get_images(numpy_features, top, j_type):
    temp = dict()
    score = list()
    
    with open(model_path, 'rb') as file:
        data = pickle.load(file)
        data = data[j_type]
        
    for image,feature in data.items():
        result = distance.cdist([numpy_features], [feature], metric)[0]
        temp[image] = result[0]
        score.append(result[0])

    output = sorted(score, key = lambda x: float(x), reverse=False)
    final_vec = []
    final_image = []

    # for number in output[0:4]:
    for number in output[:top]:
        for key,value in temp.items():
            if(number == value):
                final_vec.append(value)
                final_image.append(key)

    return final_image, final_vec
    
@app.route('/')
def upload_image():
    try: shutil.rmtree(os.path.join("static", "media"))
    except: pass
    try: shutil.rmtree(os.path.join("static", "crop"))
    except: pass
    try: shutil.rmtree("runs")
    except: pass
    
    os.mkdir(os.path.join("static", "media"))
    os.mkdir(os.path.join("static", "crop"))

    return render_template('data.html')

@app.route('/SendCoordinates', methods=['GET', 'POST'])
def SendCoordinates():
    '''
    This function is used to get coordinates while user 
    snips data from invoice. Coordinates will give us a small 
    portion of image from which text will be extracted.
    Coordinates with image name is sent to get_html()
    function.
    '''
    if request.method == 'POST':
        data = request.get_json()
            
        j_type = data["j_type"]
        Coordinates = data["values"]
        filename = data["filename"].split("/")[-1]

        img = Image.open(os.path.join('static/media', filename))
        image = img.crop((Coordinates["x"], Coordinates["y"], Coordinates["x2"], Coordinates["y2"]))
        
        cropped_image = "cropped_" + filename
        
        crop = 1
        while os.path.exists(os.path.join('static/crop', cropped_image)):
            cropped_image = str(crop) + cropped_image
            crop += 1
        
        image.save(os.path.join('static/crop', cropped_image))
        
        print("manually cropped")

        return {"image": cropped_image, "cropped": "done"}
        
    return ""

@app.route('/crop', methods = ['GET','POST'])
def crop_image():
    if request.method=='POST':
        img = request.form.get("cropped_img")
        j_type = request.form.get("j_type")

    return render_template('crop2.html', img=img, j_type=j_type)

@app.route('/input_image', methods = ['GET','POST'])
def input_image():
    if request.method=='POST':
        try:
            j_type = request.form.get("j_type")
            cropped_img = request.form.get("cropped_img")
            filename = request.form.get("filename").split("/")[-1]
            
            image = Image.open(cropped_img)
            
            print("manually cropped")
        
        except:
            j_type = request.form.get("j_type")
            file1 = request.files['file']
            
            in_mem_file = BytesIO(file1.read())
            pre_image = Image.open(in_mem_file)
            pre_image = pre_image.resize((640, 640))
            pre_image.save(os.path.join('static/media', file1.filename))
            
            # file1.save(os.path.join('static/media', file1.filename))
            filename = file1.filename

            image = Image.open(os.path.join('static/media', filename))
            results = yolo_model([image])
            crops = results.crop(save=True)

            if len(crops) > 0:
                img = crops[0]["im"][:, :, [2,1,0]]
                image = Image.fromarray(img)
                
            print("auto cropped")
                
        if image:
            width, height = image.size

            if(width != height):
                bigside = width if width > height else height

                background = Image.new('RGB', (bigside, bigside), (0, 0, 0))
                offset = (int(round(((bigside - width) / 2), 0)), int(round(((bigside - height) / 2),0)))

                background.paste(image, offset)
                background.save(os.path.join('static/crop', filename))

                print("Image has been resized !")

            else:
                print("Image is already a square, it has not been resized !")


                img_savename = os.path.join('static/crop', filename)
                image.save(img_savename)

            numpy_features = feature_extractor(os.path.join('static/crop', filename))

        else:
            numpy_features = feature_extractor(os.path.join('static/media', filename))

        temp_top = 1000

        all_images = []
        all_scores = []
        all_label_ids = []

        saved_imgs.append(filename)

        images, scores = get_images(numpy_features, temp_top, j_type)

        top = 5
        count = 0
        i = 0

        # for i in range(top):
        while(count < top):
            #for augmented data, show image from original database, not from augmented database
            img_name = images[i].split("___")[0]
            i += 1
            if str("media/"+ img_name) not in all_images:
                shutil.copy(os.path.join(database_images_path, j_type, img_name), os.path.join('static/media', img_name))

                # #without augmentation, show iamge from one folder only
                # shutil.copy(os.path.join(database_images_path,images[i]), os.path.join('static/media', images[i]))
                all_scores.append(round(100 - scores[i]*100,2))
                saved_imgs.append(img_name)

                all_images.append("media/"+ img_name)
                all_label_ids.append(img_name.split(".")[0])

                count += 1

        # #to display only one match
        # return render_template('upload-data-file.html',upload_image = "media/"+file1.filename , predicted_image = "media/"+images[0] ,per1 = sc1, labelid=images[0].split(".")[0])

        # #to display n matches
        print("return")
        return render_template(
            'upload-data-file_n_images.html', 
            upload_image = "media/"+filename,
            cropped_image = "crop/"+filename,
            top = 5, #len(all_images),
            all_images = all_images, 
            all_scores = all_scores, 
            all_label_ids = all_label_ids,
            j_type = j_type
        )

if __name__=='__main__':
    device = torch.device("cpu")
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path="model/all_exp16.pt", force_reload=True, device=device)
    
    app.run(host='0.0.0.0', port=5005, debug = True)