from flask import Flask
from flask import request
import pickle
import torchvision

import numpy as np

from load_model import load_model

import json

from torch import load
import torch

from PIL import Image
import matplotlib.pyplot as plt


app = Flask(__name__)

from flask_cors import CORS
CORS(app)
h = 256
w = h

model = load_model("model_hybrid_3x3_1.16loss.pt")

@app.route('/', methods=["GET", "POST"])
def hello_world():
    if request.method == "GET":
        return 'Please use POST and submit the image and mask as json'
    if request.method == "POST":
        # print(request.data)
        img, mask = get_image_and_mask_from(request.data)

        img = np.array(img, dtype=np.uint8)
        plt.imsave("input.png", img)
        mask = np.array(mask, dtype=np.uint8, ndmin=2)
        mask = np.reshape(mask, (h, w, 1))

        print(np.unique(mask.reshape(256*256)))


        mask = np.negative(mask-1)
        plt.imsave("mask.png", np.reshape(mask.astype("float"), (h,w)))
        mask = torch.Tensor(np.transpose(mask, (2, 0, 1)))


        transformed_image = transform(img)
        print(mask.shape)
        print(transformed_image.shape)
        input_ = torch.cat((transformed_image, mask), 0)
        input_ = torch.reshape(input_, (1, 4, h, w))
        print(input_.shape)
        print(input_)
        #print(input_)
        np_pred = (model(input_)*255).detach().numpy()
        print(np_pred)
        print(np_pred.shape)
        np_pred = np.transpose(np_pred, (0,2,3,1)) 
        print(np_pred.shape)
        np_pred = np_pred.reshape((256,256,3))
        print(np_pred.shape)

        np_pred = np.floor(np_pred)
        print(np_pred)
        plt.imsave("prediction_.jpg", np_pred.astype('uint8'))
        #np_pred = np.absolute(np_pred-255)
        pred = np_pred.astype('uint8').tolist()

        plt.imsave("prediction.jpg", pred)

        #print(pred)

        return json.dumps(get_data_from_image(pred))
    else:
        return "Invalid HTTP Method"


def get_data_from_image(image):
    # print(image)
    image_height = 256
    image_width = image_height
    image_data = []
    for y in range(image_height):
         for x in range(image_width):
            red = image[y][x][0]
            green = image[y][x][1] 
            blue = image[y][x][2]
            alpha = 255
            image_data.append(red)
            image_data.append(green)
            image_data.append(blue)
            image_data.append(alpha)
    return image_data


def get_image_and_mask_from(json_string):
    json_string = str(json_string, encoding="utf-8")

    json_string = json_string.replace("null", "0")

    obj = json.loads(json_string)
    damaged_img = get_image_from_data(obj['damagedImage'])
    with open("img_transmitted.txt", "w") as f:
        print(damaged_img, file=f)
    mask = obj['mask']

    for y in range(256):
        for x in range(256):
            counter = 0
            for p in damaged_img[y][x]:
                if p == 0:
                    counter += 1
            if counter == 3:
                mask[y][x] = 1.

    for m in mask:
        if not len(m) == 256:
            print(m)
            print("RIP:" + str(len(m)))

    return damaged_img, mask


def get_image_from_data(data):
    image_height = 256
    image_width = image_height
    image = np.ndarray((256,256,3), dtype=np.uint8)
    for y in range(image_height):
         for x in range(image_width):
            red = data[((image_width * y) + x) * 4]
            green = data[((image_width * y) + x) * 4 + 1]
            blue = data[((image_width * y) + x) * 4 + 2]
            alpha = data[((image_width * y) + x) * 4 + 3]
            
            image[y][x][0] = red
            image[y][x][1] = green
            image[y][x][2] = blue

    return image

def transform(image):
    image = torchvision.transforms.ToTensor()(image)
    return image
