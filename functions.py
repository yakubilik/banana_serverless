import torch
from models.psp import pSp
from argparse import Namespace
import numpy as np
import torchvision.transforms as transforms
import logging


from datasets.augmentations import AgeTransformer
from utils.common import tensor2im
from models.psp import pSp
from PIL import Image


from PIL import Image
from pillow_heif import register_heif_opener
from io import BytesIO
import base64
register_heif_opener()

def convert_to_base64(image):
    im_file = BytesIO()
    image.save(im_file, format="JPEG")
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    im_b64 = base64.b64encode(im_bytes)
    return im_b64

def image_from_base64(bs4_image):
    print("Image alınıyor..")
    bs4_image = bs4_image.encode("utf-8")
    im = Image.open(BytesIO(base64.b64decode(bs4_image)))
    return im


  
def load_model():
    print("Model Yükleniyor!!")
    EXPERIMENT_DATA_ARGS = {
        "ffhq_aging": {
            "model_path": "../pretrained_models/sam_ffhq_aging.pt",
            "image_path": "notebooks/images/866.jpg",
            "transform": transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        }
    }


    model_path = "./pretrained_models/sam_ffhq_aging.pt"

    ckpt = torch.load(model_path, map_location='cpu')

    opts = ckpt['opts']

    opts['checkpoint_path'] = model_path

    opts = Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cuda()
    return net


def run_alignment(image):
    import dlib
    from scripts.align_all_parallel import align_face
    predictor = dlib.shape_predictor("./pretrained_models/shape_predictor_68_face_landmarks.dat")
    image.save("align.jpg")

    aligned_image = align_face(filepath="./align.jpg", predictor=predictor) 
    print("aligned")

    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image 




def run_on_batch(inputs, net):
    print("Run on Batch")
    result_batch = net(inputs.to("cuda").float(), randomize_noise=False, resize=False)
    return result_batch
    


def predict_model(image,target_age:int,net):
    print("Predicting!!!")
    img_transforms = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    target_ages = [target_age]

    age_transformers = [AgeTransformer(target_age=age) for age in target_ages]
    images_list = []
    original_image = image_from_base64(image)
    original_image.resize((256, 256))
    print("resized!")


    for i in range(4):
        try:
            aligned_image = run_alignment(original_image)
            print("aligned!")
            break
        except:
            original_image = original_image.rotate(-90)
            aligned_image = run_alignment(original_image)
            break

    input_image = img_transforms(aligned_image)
    print("transform!")


    for age_transformer in age_transformers:
        print(f"Running on target age: {age_transformer.target_age}")
        with torch.no_grad():
            input_image_age = [age_transformer(input_image.cpu()).to('cuda')]
            input_image_age = torch.stack(input_image_age)
            result_tensor = run_on_batch(input_image_age, net)[0]
            result_image = tensor2im(result_tensor)
            bs64_image = convert_to_base64(result_image)
            decoded = bs64_image.decode('utf-8')
            images_list.append(decoded)

    return images_list 


