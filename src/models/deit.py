from transformers import DeiTForImageClassification, AutoImageProcessor
from torchvision import transforms
import torch.nn.functional as nnf
from PIL import Image
import torch


class DeitFinetuner:

    def __init__(self, device, num_classes, path = None):
        self.model_name = "facebook/deit-tiny-distilled-patch16-224"
        self.num_classes = num_classes
        self.device = device

        self.image_processor = AutoImageProcessor.from_pretrained(self.model_name, use_fast=False)

        if path == None:
            self.model = DeiTForImageClassification.from_pretrained(self.model_name, num_labels=num_classes)
        else:
            self.model = DeiTForImageClassification.from_pretrained(path)

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

            outputs = self.model(inputs).logits
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

    def infer(self, path, k = -1, id2label = None):
        image = Image.open(path)
        input = self.image_processor(images=image, return_tensors="pt")
        self.model.eval()
        input = input.to(self.device)
        logits = self.model(**input).logits
        prob = nnf.softmax(logits, dim=1)

        if k == -1:
            k = self.num_classes

        values, indices = prob.topk(k)
        values = values[0]
        indices = indices[0]

        if id2label == None:
            predictions = [
                {"label": i.item(), "score": v.item()}
                for i, v in zip(indices, values)
            ]
        else:
            predictions = [
                {"label": id2label[i.item()], "score": v.item()}
                for i, v in zip(indices, values)
            ]

        return predictions

    def validate(self, loader, loss_fn):
        v_loss = self.valid_loss(loader,loss_fn)
        v_acc1 = self.accuracy(loader,1)
        v_acc5 = self.accuracy(loader,5)
        return v_loss, v_acc1, v_acc5

    def valid_loss(self, loader, loss_fn):
        self.model.eval()
        total_valid_loss = 0
        for data in loader:
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs).logits
            loss = loss_fn(outputs, labels)
            total_valid_loss += loss.item()

        #avr_loss
        return total_valid_loss / (len(loader) * loader.batch_size)

    def accuracy(self, loader, k):
        self.model.eval()
        r = 0

        with torch.no_grad():
            for vdata in loader:
                inputs, labels = vdata
                inputs = inputs.to(self.device)
                outputs = self.model(inputs).logits
                m = len(outputs)

                for i in range(m):
                    label = labels[i].item()
                    prob = nnf.softmax(outputs[i], dim=-1)
                    _, indices = prob.topk(k)

                    if k == 1:
                        indices = indices.item()
                        r += 1 if label == indices else 0
                    else:
                        indices = indices.tolist()
                        r += 1 if label in indices else 0

        return r / (len(loader) * loader.batch_size)

    def save(self,path):
        self.model.save_pretrained(path)