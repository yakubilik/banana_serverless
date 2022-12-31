# Do not edit if deploying to Banana Serverless
# This file is boilerplate for the http server, and follows a strict interface.

# Instead, edit the init() and inference() functions in app.py
import logging
from sanic import Sanic, response
import app as user_src
from flask import Flask, request, jsonify
import json
import torch
import ast
# We do the model load-to-GPU step on server startup
# so the model object is available globally for reuse
user_src.init()
# Create the http server app
server = Flask(__name__)

def send_log(req):
    import requests
    import json

    url = "http://192.168.0.148:8080/"

    payload = json.dumps(req)
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)


# Healthchecks verify that the environment is correct on Banana Serverless
@server.route("/healtcheck")
def health_check():
    # dependency free way to check if GPU is visible
    gpu = torch.cuda.is_available()
    return_json = {"state": "healthy", "gpu": gpu}

    return json.dumps(return_json)

# Inference POST handler at '/' is called for every http call from Banana
@server.route("/",methods=["GET","POST"])
def inference():
    print('request geldi')
    print(request)
    js = request.json
    send_log(js)
    inputs = js.get("modelInputs")
    image = inputs["image"]
    target_age = inputs["target_age"]
    print("Target_list",target_age)
    
    print("Target_list",type(target_age))


    model_inputs = {"image":image,"target_age":target_age}

    output = user_src.inference(model_inputs)
    
    print("predict edildi")
    print(output)
    send_log({"oldImage": output})
    return {"oldImage": output} 


if __name__ == '__main__':
    server.run(host='0.0.0.0', port=8000)

