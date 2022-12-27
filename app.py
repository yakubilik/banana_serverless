from transformers import pipeline
import torch
import sys 
from functions import load_model, predict_model
import json

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    model = load_model()

    def convert_to_base64(image):
        im_file = BytesIO()
        image.save(im_file, format="JPEG")
        im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
        im_b64 = base64.b64encode(im_bytes)
        return im_b64

    def image_from_base64(bs4_image):
        im = Image.open(BytesIO(base64.b64decode(bs4_image)))
        return im


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    try:
        image = model_inputs.get("image")
        target_age = model_inputs.get("target_age")
        result = predict_model(image=image,target_age=target_age,net=model)

        response_dict = {}

        return result[0]
    except:
        return False

