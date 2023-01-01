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


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    image = model_inputs.get("image")
    target_age = model_inputs.get("target_age")
    result = predict_model(image=image,target_age=target_age,net=model)

    response_dict = {}

    return result[0]

