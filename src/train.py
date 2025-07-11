import os
import time
import jsonpickle
import torch
import datetime
from safetensors.torch import save_file
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from models.effnet import EffNet
from models.levit import LeVit
from models.deit import Deit
from src.other.earlyStopper import EarlyStopper

from src.other.stats import Stats

# Configuration
num_classes = 525  # your number of output classes
batch_size = 16
num_epochs = 50

torch.manual_seed(7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Deit;LeVit;EffNet
model = Deit(device, 525, pretrained=True)#,"distilled_tiny"

# Add folder structure for trained Models
folder_name = f"birds_{model.short_name}" #_npt
folder_path = f"../networks/{folder_name}"
weights_path = f"{folder_path}/weights"

early_stopper = EarlyStopper(patience=3,min_delta=.001)

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

if not os.path.exists(weights_path):
    os.makedirs(weights_path)

#Load training set
train_folder = "D:\\Datasets\\bird-species-dataset\\data\\train"
train_dataset = datasets.ImageFolder(root=train_folder, transform=model.transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#Load validation set
valid_folder = "D:\\Datasets\\bird-species-dataset\\data\\valid"
valid_dataset = datasets.ImageFolder(root=valid_folder, transform=model.transform)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

# stores the idx to class as json
class_names = {idx: cls for cls, idx in train_dataset.class_to_idx.items()}
json_object = jsonpickle.encode(class_names)

# save id2label as json
with open(f"{folder_path}/idx2classes.json", "w") as file:
    file.write(json_object)

# Optimizer and loss
optimizer = optim.Adam(model.model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

# Training loop
stats_valid_list = []
stats_train_list = []

#save stats as csv
with open(f"{folder_path}/logs_train.csv","w") as t, open(f"{folder_path}/logs_valid.csv","w") as v:
    t.write(Stats.csv_head())
    t.flush()
    v.write(Stats.csv_head())
    v.flush()

    training_start_time = time.time()

    for epoch in range(num_epochs):
        print(datetime.datetime.now())
        epoch_start_time = time.time()

        avr_train_loss = model.train_epoch(train_loader, loss_fn, optimizer)
        train_time = time.time() - epoch_start_time

        #keras.io/getting_started/faq/#why-is-my-training-loss-much-higher-than-my-testing-loss
        stats_train = model.stats(train_loader, loss_fn, 'micro')
        stats_train.epoch = epoch
        stats_train.seconds = train_time
        t.write(stats_train.csv())
        t.flush()
        print('[valid]', stats_train)


        stats_valid = model.stats(valid_loader, loss_fn, 'micro')
        stats_valid.epoch = epoch
        stats_valid.seconds = train_time
        v.write(stats_valid.csv())
        v.flush()
        print('[train]', stats_valid)

        epoch_time_str = (time.time() - epoch_start_time) /60
        print('[Epoch Time]',epoch_time_str)

        if os.path.exists(weights_path):
            model.save(f"{weights_path}/{epoch + 1}.safetensors")

        if early_stopper.early_stop(stats_valid.avr_loss):
            break

        print()

train_time_str = time.time() - training_start_time
print(f"Training Duration: {train_time_str}")

# Save model weights as safetensors
model.save(f"{weights_path}/weights_final.safetensors")
