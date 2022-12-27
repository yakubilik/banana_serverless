# Do not edit if deploying to Banana Serverless
# This file is boilerplate for the http server, and follows a strict interface.

# Instead, edit the init() and inference() functions in app.py

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
    print('geldi')
    print(request)
    try:
        js = request.json
        inputs = js.get("modelInputs")
        image = inputs["image"]
        target_list = inputs["target_list"]
        target_list = ast.literal_eval(target_list)
        print("Target_list",target_list)
        print("Target_list",type(target_list))


        model_inputs = {"image":image,"target_list":target_list}

        output = user_src.inference(model_inputs)
        print("output döndü")
        print(output)

        return output


    except Exception as e:
        return jsonify(patladi=e)


if __name__ == '__main__':
    server.run(host='0.0.0.0', port=8000)

