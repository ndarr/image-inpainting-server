from flask import Flask
from flask import request
import pickle
import torchvision

import numpy as np

from load_model import load_model

import json

from torch import load
import torch


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
        # print(img)
        # print(mask)
        # img = np.zeros((h, w, 3), dtype=np.uint8)
        # mask = np.zeros((h, w, 1), dtype=np.uint8)

        img = np.array(img, dtype=np.uint8)
        mask = np.array(mask, dtype=np.uint8, ndmin=2)
        mask = np.reshape(mask, (h, w, 1))
        print(mask.shape)
        mask = torch.Tensor(np.transpose(mask, (2, 0, 1)))


        transformed_image = transform(img)
        print(mask.shape)
        print(transformed_image.shape)
        input_ = torch.cat((transformed_image, mask), 0)
        input_ = torch.reshape(input_, (1, 4, h, w))
        #print(input_)
        np_pred = (model(input_)*255).detach().numpy()
        print(np_pred.shape)
        pred = np.floor(np_pred.reshape((256, 256, 3))).astype('uint8').tolist()
        #print(pred)

        return json.dumps(pred)
    else:
        return "Invalid HTTP Method"

def get_image_and_mask_from(json_string):
    json_string = str(json_string, encoding="utf-8")

    json_string = json_string.replace("null", "0")

    obj = json.loads(json_string)
    damaged_img = obj['damagedImage']
    mask = obj['mask']

    for m in mask:
        if not len(m) == 256:
            print(m)
            print("RIP:" + str(len(m)))

    return damaged_img, mask


def transform(image):
    image = torchvision.transforms.ToTensor()(image)
    return image
