import torch
from torchvision import transforms
import timm
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from timeit import default_timer as timer

_transform=transforms.Compose([
                       transforms.Resize((224,224)),
                       transforms.ToTensor(),
                       transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

def read_img_file(f):
    img = Image.open(f)
    img=img.convert('RGB') 
    return img

models = ["tf_efficientnetv2_m_in21k", "vit_base_patch16_224_in21k", "resnetv2_50x1_bitm_in21k"]

with open("./imagenet21k_wordnet_lemmas.txt") as f:
    classes = [line.strip() for line in f.readlines()]
    
folders = os.listdir("./dataset/")

for model_name in models:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device}")
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    model.to(device)
    tp = 0
    tp_5 = 0
    _time = 0
    for folder_name in tqdm(folders):    
        files = os.listdir("./dataset/"+folder_name)
        for file_name in files:
            img = read_img_file("./dataset/"+folder_name+"/"+file_name)
            with torch.no_grad():
                img = _transform(img)
                img = torch.unsqueeze(img, 0)
                img = img.to(device)
                start = timer()
                x = model(img)
                end = timer()
                _time+=(end - start)
                x = torch.nn.functional.softmax(x, dim=1) * 100
                top_5 = torch.topk(x, 5)
            preds = []
            for x in top_5.indices[0]:
                preds.append(classes[int(x)])
            # print(preds)
            if folder_name in preds[0]:
                tp+=1
            for pred in preds:
                if folder_name in pred:
                    tp_5+=1
                    break
    print(f" acc5 = {tp_5/50} acc1 = {tp/50}")
    print(f"time: {_time}")