import time
import jsonpickle
import torch
from safetensors.torch import save_file
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from models.effnet import EffNetFinetuner

# Configuration
model_name = "facebook/deit-base-distilled-patch16-224"
num_classes = 525  # your number of output classes
batch_size = 16
num_epochs = 5
device_str = "cuda" if torch.cuda.is_available() else "cpu"
train_folder = "D:\\Datasets\\bird-species-dataset\\data\\train"
valid_folder = "D:\\Datasets\\bird-species-dataset\\data\\valid"
device = torch.device(device_str)

ft = EffNetFinetuner(device,525)#,"distilled_tiny"
transform = ft.transform
model = ft.model

#Load training set
train_dataset = datasets.ImageFolder(root=train_folder, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#Load validation set
valid_dataset = datasets.ImageFolder(root=valid_folder, transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

# stores the idx to class as json
class_names = {idx: cls for cls, idx in train_dataset.class_to_idx.items()}
json_object = jsonpickle.encode(class_names)

with open('../idx2classes.json', 'w') as file:
    file.write(json_object)

# Optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

# Training loop
training_start_time = time.time()

for epoch in range(num_epochs):
    epoch_start_time = time.time()

    avr_train_loss = ft.train_epoch(train_loader,loss_fn,optimizer)
    avr_valid_loss = ft.validate(train_loader,loss_fn)

    epoch_time_str = time.time() - epoch_start_time
    print(f"Epoch {epoch+1}/{num_epochs} - Loss train: {avr_train_loss:.4f} valid: {avr_valid_loss:.4f} - Duration: {epoch_time_str}")

train_time_str = time.time() - training_start_time
print(f"Training Duration: {train_time_str}")

# Save finetuned model
ft.save("birds_effnet_5-epochs_2.safetensors")
