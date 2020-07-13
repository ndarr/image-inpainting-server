import json
import pickle
import numpy as np
import torch
from torch import load
import torchvision
from flask import Flask
from flask import request
from flask_cors import CORS
from PIL import Image
import matplotlib.pyplot as plt

from load_model import load_model

app = Flask(__name__)

CORS(app)
h = 256
w = h

"""
This flask application provides a REST endpoint for our image inpainting neural network
"""

# load model from checkpoint
model = load_model("model_hybrid_3x3_1.16loss.pt")

@app.route('/', methods=["GET", "POST"])
def get_prediction():
    """
    Prediction endpoint (POST) which takes image and mask and returns inpainted image
    """

    if request.method == "GET":
        return 'Please use POST and submit the image and mask as json'

    if request.method == "POST":
        
        # get inputs
        img, mask = get_image_and_mask_from(request.data)
        img = np.array(img, dtype=np.uint8)

        # prepare mask for model
        mask = np.array(mask, dtype=np.uint8, ndmin=2)
        mask = np.reshape(mask, (h, w, 1))
        mask = np.negative(mask-1)
        mask = torch.Tensor(np.transpose(mask, (2, 0, 1)))

        # prepare image for model and concatenate with mask
        transformed_image = transform(img)
        input_ = torch.cat((transformed_image, mask), 0)
        input_ = torch.reshape(input_, (1, 4, h, w))

        # get model prediction
        np_pred = (model(input_)*255).detach().numpy()

        # prepare prediction to send back to client
        np_pred = np.transpose(np_pred, (0,2,3,1)) 
        np_pred = np_pred.reshape((256,256,3))
        np_pred = np.floor(np_pred)
        pred = np_pred.astype('uint8').tolist()

        return json.dumps(get_data_from_image(pred))
        
    else:
        return "Invalid HTTP Method"


def get_data_from_image(image):
    """
    Takes in inpainted image prediction as multidimensional list (RGB channels), extracts the data, and writes it into flat list
    """

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
    """
    Takes in JSON string from client and outputs the damaged image and mask
    """

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

    return damaged_img, mask


def get_image_from_data(data):
    """
    Takes in image data as 1-d list, distributes pixels among respective color channels, and returns 3-d RGB image
    """

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
    """
    Takes in image as numpy ndarray (RGB channels), transforms it to tensor, and returns image tensor
    """

    image = torchvision.transforms.ToTensor()(image)
    return image
