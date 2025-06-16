import timm
import torch
from PIL import Image
from transformers import LevitForImageClassification, LevitImageProcessor
from torchvision import datasets, transforms
from safetensors.torch import save_file

class LeVitFinetuner:

    def __init__(self,device, num_classes, path = None):
        self.model_name = "facebook/levit-128S"
        self.device = device
        self.num_classes = num_classes
        self.image_processor = LevitImageProcessor.from_pretrained(self.model_name)

        if path == None:
            self.model = timm.create_model("timm/levit_128s.fb_dist_in1k",pretrained=True,num_classes=num_classes)
        else:
            pretrained_config_overlay=dict(file=path)
            self.model = timm.create_model("timm/levit_128s.fb_dist_in1k",pretrained=True,num_classes=num_classes,
            pretrained_cfg_overlay=pretrained_config_overlay)

        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.image_processor.image_mean, std=self.image_processor.image_std),
        ])

    def train_epoch(self, loader, loss_fn, optimizer):
        self.model.train()
        total_train_loss = 0
        for batch in loader:
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(inputs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

    def infer(self, path, k = -1):
        self.model.eval()

        #labels = self.model.pretrained_cfg['label_names']

        image = Image.open(path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        outputs = self.model(tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

        if k == -1:
            k = self.num_classes

        values, indices = probs.topk(k)

        predictions = [
            {"label": i.item(), "score": v.item()}
            for i, v in zip(indices, values)
        ]

        return predictions

    def validate(self, loader, loss_fn):
        self.model.eval()
        total_valid_loss = 0
        for data in loader:
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
            loss = loss_fn(outputs,labels)
            total_valid_loss += loss.item()

    def valid_loss(self, loader, loss_fn):
        return

    def accuracy(self, loader, k):
        return

    def save(self,path):
        state_dict = self.model.state_dict()
        save_file(state_dict, path) #f"{path}\\levit_5-epochs.safetensors"