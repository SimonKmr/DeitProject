import os
import time
import jsonpickle
import torch
from safetensors.torch import save_file
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from models.effnet import EffNet
from models.levit import LeVit
from models.deit import Deit

from src.other.stats import Stats

# Configuration
num_classes = 525  # your number of output classes
batch_size = 16
num_epochs = 10

train_folder = "D:\\Datasets\\bird-species-dataset\\data\\train"
valid_folder = "D:\\Datasets\\bird-species-dataset\\data\\valid"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EffNet(device, 525)#,"distilled_tiny"


#Load training set
train_dataset = datasets.ImageFolder(root=train_folder, transform=model.transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#Load validation set
valid_dataset = datasets.ImageFolder(root=valid_folder, transform=model.transform)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

# stores the idx to class as json
class_names = {idx: cls for cls, idx in train_dataset.class_to_idx.items()}
json_object = jsonpickle.encode(class_names)

# Add folder structure for trained Models
folder_name = f"birds_{model.short_name}"
folder_path = f"../networks/{folder_name}"

if not os.path.exists(folder_path):
    os.makedirs(f"{folder_path}")

# save id2label as json
with open(f"{folder_path}/idx2classes.json", "w") as file:
    file.write(json_object)

# Optimizer and loss
optimizer = optim.Adam(model.model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

# Training loop
training_start_time = time.time()
stats_list = []
for epoch in range(num_epochs):
    epoch_start_time = time.time()

    avr_train_loss = model.train_epoch(train_loader, loss_fn, optimizer)
    train_time = time.time() - epoch_start_time
    stats = model.stats(train_loader, loss_fn, "weighted")
    stats.epoch = epoch
    stats.seconds = train_time

    stats_list.append(stats)

    if os.path.exists(folder_path):
        model.save(f"{folder_path}/weights_{epoch + 1}.safetensors")

    epoch_time_str = (time.time() - epoch_start_time) /60
    print(stats)

train_time_str = time.time() - training_start_time
print(f"Training Duration: {train_time_str}")


#save stats as csv
with open(f"{folder_path}/logs.csv","w") as f:
    Stats.csv_head()
    for e in stats_list:
        f.write(e.csv())

# Save model weights as safetensors
model.save(f"{folder_path}/weights_final.safetensors")
