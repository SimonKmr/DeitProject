import timm
from fontTools.pens.basePen import NullPen
from transformers import DeiTForImageClassification, AutoImageProcessor
from torchvision import transforms
import torch.nn.functional as nnf
from PIL import Image
from torcheval.metrics import MulticlassF1Score, MulticlassRecall, MulticlassPrecision, MulticlassAccuracy
import torch
from torch.utils.flop_counter import FlopCounterMode
from safetensors.torch import save_file, load_file
from src.other.hooks import HookingManager
from src.other.stats import Stats


class Deit:
    def __init__(self,device, num_classes, path = None):
        self.short_name = "deit"
        self.model_name = "facebook/deit-tiny-distilled-patch16-224"
        self.device = device
        self.num_classes = num_classes
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_name, use_fast=False)

        if path == None:
            self.model = timm.create_model("timm/deit_tiny_patch16_224.fb_in1k",pretrained=True,num_classes=num_classes)
        else:
            state_dict= load_file(path)
            self.model = timm.create_model("timm/deit_tiny_patch16_224.fb_in1k",num_classes=num_classes)
            self.model.load_state_dict(state_dict)

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

        return total_train_loss / len(loader)

    def infer(self,input,k = -1):
        self.model.eval()

        if input is str:
            image = Image.open(input).convert('RGB')
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

        elif input is tuple[int,int,int,int]:
            return self.model(input)

        elif input is torch.utils.data.DataLoader:
            result = []
            with torch.no_grad():
                for data in input:
                    inputs, _ = data
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs)
                    result.append(outputs)
            return result



    def save(self,path):
        state_dict = self.model.state_dict()
        save_file(state_dict, path)

    def stats(self, loader, loss_fn, average = None):
        loss = self.valid_loss(loader, loss_fn)
        acc1, acc5, f1_score, precision, recall = self.metrics(loader,average)
        return Stats(loss, acc1, acc5, f1_score, precision, recall)

    def valid_loss(self, loader, loss_fn):
        self.model.eval()
        total_valid_loss = 0
        for data in loader:
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
            loss = loss_fn(outputs, labels)
            total_valid_loss += loss.item()

        #avr_loss
        return total_valid_loss / (len(loader) * loader.batch_size)

    def metrics(self, loader, average = None):
        self.model.eval()
        precision = MulticlassPrecision(num_classes=self.num_classes, average=average)
        recall = MulticlassRecall(num_classes=self.num_classes, average=average)
        f1_score = MulticlassF1Score(num_classes=self.num_classes, average=average)
        acc1 = MulticlassAccuracy(num_classes=self.num_classes, k=1)
        acc5 = MulticlassAccuracy(num_classes=self.num_classes, k=5)
        with torch.no_grad():
            for data in loader:
                inputs, targets = data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                precision.update(outputs, targets)
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

    def flops(self):
        flop_counter = FlopCounterMode(display=False, depth=None)
        self.model.eval()
        input = (1,3,224,224)
        input = torch.randn(input)
        input = input.to(self.device)
        with flop_counter:
            self.model(input)
        total_flops = flop_counter.get_total_flops()
        return total_flops

    def infer_hooked(self, path : str, layer_index = 0):
        self.model.eval()

        image = Image.open(path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0)

        hm = HookingManager()
        model_children = list(self.model.to('cpu').children())
        model_children[layer_index].register_forward_hook(hm.hook)

        _ = self.model(tensor)

        self.model.to(self.device)

        return hm.result