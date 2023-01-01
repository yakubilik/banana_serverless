from PIL import Image
from functions import predict_model,load_model,image_from_base64
import json
with open("data1.json") as f:
    content = f.read()
    content_j = json.loads(content)
    image = content_j.get("modelInputs").get("image")

target_age = 70
model = load_model()

result = predict_model(image=image,target_age=target_age,net=model)

res_img = image_from_base64(result[0])

res_img.save("result_rotated.png")