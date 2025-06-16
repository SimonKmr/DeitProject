import torch
import json
import torch.nn as nn
from models.deit import DeitFinetuner
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(3)

# Configuration
device_str = "cuda" if torch.cuda.is_available() else "cpu"
print("running on",device_str)
device = torch.device(device_str)
batch_size = 16

df = DeitFinetuner(device, 525, "../networks/birds_deit-dt_5-epochs")
path = "D:\\Datasets\\bird-species-dataset\\data\\valid"
dataset = datasets.ImageFolder(root=path, transform=df.transform)
loader = DataLoader(dataset, batch_size=batch_size)

predictions = df.infer("C:\\Users\\Simon\\Desktop\\blaumeise_1.jpg",5)
print(predictions)

loss, acc1, acc5 = df.validate(loader,nn.CrossEntropyLoss())
print(f"loss: {loss:.4}, acc1: {acc1:.4}, acc5: {acc5:.4}")

with open("../idx2classes.json") as f:
    id2label = json.load(f)

# show top 5 propabilities
#print("Predicted class:", df.model.config.id2label[predictions[0].label])
for i in predictions:
    label = id2label[str(i["label"])]
    prob = i["score"]
    print(f'{label}: {prob:.4}')