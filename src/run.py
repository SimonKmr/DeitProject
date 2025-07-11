import torch
import json

from models.deit import Deit
from models.effnet import EffNet
from models.levit import LeVit

from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn

torch.manual_seed(3)

# Configuration
device_str = "cuda" if torch.cuda.is_available() else "cpu"
print("running on",device_str)
device = torch.device(device_str)
batch_size = 16
average = "weighted" # https://docs.pytorch.org/torcheval/stable/generated/torcheval.metrics.functional.multiclass_f1_score.html#torcheval.metrics.functional.multiclass_f1_score
model_selection = "deit-npt"

model = None
if model_selection == "deit":
    model = Deit(device, 525, "../networks/birds_deit/weights_final.safetensors")
elif model_selection == "deit-npt":
    model = Deit(device, 525, "../networks/birds_deit_npt/weights_final.safetensors")
elif model_selection == "levit":
    model = LeVit(device, 525, "../networks/birds_levit/weights_final.safetensors")
elif model_selection == "effnet":
    model = EffNet(device, 525, "../networks/birds_effnet/weights_final.safetensors")
#path = "D:\\Datasets\\bird-species-dataset\\data\\valid"
#dataset = datasets.ImageFolder(root=path, transform=df.transform)
#loader = DataLoader(dataset, batch_size=batch_size)

#stats = df.stats(loader,nn.CrossEntropyLoss(),average)
#print(stats)

with open("../networks/birds_levit/idx2classes.json") as f:
    id2label = json.load(f)

path ="C:\\Users\\Simon\\Desktop\\birds\\Goulds Toucanet.jpg"
predictions = model.infer(path,5)#blaumeise_1.jpg
print(predictions)

flops = model.flops()
print(flops)

# show top 5 propabilities
print("Predicted class:", id2label[str(predictions[0]["label"])])
for i in predictions:
    label = id2label[str(i["label"])]
    prob = i["score"]
    print(f'{label}: {prob:.4}')

#Load validation set
valid_folder = "D:\\Datasets\\bird-species-dataset\\data\\valid"
valid_dataset = datasets.ImageFolder(root=valid_folder, transform=model.transform)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

loss_fn = nn.CrossEntropyLoss()
stats_valid = model.stats(valid_loader, loss_fn, "weighted")
print(stats_valid)