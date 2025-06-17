from transformers import DeiTForImageClassification, AutoImageProcessor
from torchvision import transforms
import torch.nn.functional as nnf
from PIL import Image
from torcheval.metrics import MulticlassF1Score, MulticlassRecall, MulticlassPrecision, MulticlassAccuracy
import torch

from src.other.stats import Stats


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

    def save(self,path):
        self.model.save_pretrained(path)

    def validate(self, loader, loss_fn):
        loss = self.valid_loss(loader,loss_fn)
        acc1, acc5, f1_score, precision, recall = self.metrics(loader)
        return Stats(loss, acc1, acc5, f1_score, precision, recall)

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

    def metrics(self, loader):
        self.model.eval()
        precision = MulticlassPrecision(num_classes=self.num_classes)
        recall = MulticlassRecall(num_classes=self.num_classes)
        f1_score = MulticlassF1Score(num_classes=self.num_classes)
        acc1 = MulticlassAccuracy(num_classes=self.num_classes, k=1)
        acc5 = MulticlassAccuracy(num_classes=self.num_classes, k=5)
        with torch.no_grad():
            for data in loader:
                inputs, targets = data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs).logits
                precision.update(outputs,targets)
                recall.update(outputs, targets)
                f1_score.update(outputs, targets)
                acc1.update(outputs, targets)
                acc5.update(outputs, targets)

        res_precision = precision.compute().item()
        res_recall = recall.compute().item()
        res_f1_score = f1_score.compute().item()
        res_acc1 = acc1.compute().item()
        res_acc5 = acc5.compute().item()
        return res_acc1, res_acc5, res_f1_score, res_precision, res_recall