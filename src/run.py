import torch
import json

from models.deit import DeitFinetuner
from models.effnet import EffNetFinetuner
from models.levit import LeVitFinetuner

torch.manual_seed(3)

# Configuration
device_str = "cuda" if torch.cuda.is_available() else "cpu"
print("running on",device_str)
device = torch.device(device_str)
batch_size = 16
average = "weighted" # https://docs.pytorch.org/torcheval/stable/generated/torcheval.metrics.functional.multiclass_f1_score.html#torcheval.metrics.functional.multiclass_f1_score

models = {
    "deit" : DeitFinetuner(device, 525, "../networks/birds_deit-dt_5-epochs"),
    "effnet" : EffNetFinetuner(device, 525, "../networks/birds_effnet_5-epochs.safetensors"),
    "levit" : LeVitFinetuner(device, 525, "../networks/birds_levit_5-epochs.safetensors")
}

df = models["levit"]
#path = "D:\\Datasets\\bird-species-dataset\\data\\valid"
#dataset = datasets.ImageFolder(root=path, transform=df.transform)
#loader = DataLoader(dataset, batch_size=batch_size)

#stats = df.stats(loader,nn.CrossEntropyLoss(),average)
#print(stats)

with open("../idx2classes.json") as f:
    id2label = json.load(f)

predictions = df.infer("C:\\Users\\Simon\\Desktop\\blaumeise_1.jpg",5)
print(predictions)

# show top 5 propabilities
print("Predicted class:", id2label[str(predictions[0]["label"])])
for i in predictions:
    label = id2label[str(i["label"])]
    prob = i["score"]
    print(f'{label}: {prob:.4}')