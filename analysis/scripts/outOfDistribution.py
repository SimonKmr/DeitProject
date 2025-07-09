import torch
import json
import sys
import os
import glob
from torchvision import datasets
from torch.utils.data import DataLoader

sys.path.insert(0, '../../src/models')

from deit import Deit
from effnet import EffNet
from levit import LeVit

torch.manual_seed(3)

# Configuration
device_str = "cuda" if torch.cuda.is_available() else "cpu"
print("running on",device_str)
device = torch.device(device_str)
batch_size = 1
average = "weighted" # https://docs.pytorch.org/torcheval/stable/generated/torcheval.metrics.functional.multiclass_f1_score.html#torcheval.metrics.functional.multiclass_f1_score
model_selection = "effnet"
models = {
    "deit" : Deit(device, 525, "../../networks/birds_deit/weights_final.safetensors"),
    "effnet" : EffNet(device, 525, "../../networks/birds_effnet/weights_final.safetensors"),
    "levit" : LeVit(device, 525, "../../networks/birds_levit/weights_final.safetensors")
}

model = models[model_selection]

dir = "C:\\Users\\Simon\\Desktop\\birds\\set\\"
imgs = os.listdir(dir)

with open(f"../../networks/birds_{model_selection}/idx2classes.json") as f:
    id2label = json.load(f)

with open(f"../../networks/birds_{model_selection}/outOfDistribution.csv",'w') as w:


    for img in imgs:
        path_to_img = dir+img
        predictions = model.infer(path_to_img)

        i = img.replace('.jpg','')

        l1 = id2label[str(predictions[0]["label"])]
        l2 = id2label[str(predictions[1]["label"])]
        l3 = id2label[str(predictions[2]["label"])]
        l4 = id2label[str(predictions[3]["label"])]
        l5 = id2label[str(predictions[4]["label"])]

        s1 = predictions[0]["score"]
        s2 = predictions[1]["score"]
        s3 = predictions[2]["score"]
        s4 = predictions[3]["score"]
        s5 = predictions[4]["score"]
        csv = f'{i};{l1};{s1};{l2};{s2};{l3};{s3};{l4};{s4};{l5};{s5}\n'
        print(csv)

        w.write(csv)