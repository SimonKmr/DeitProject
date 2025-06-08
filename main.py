from transformers import AutoImageProcessor, DeiTForImageClassificationWithTeacher

import torch, torchvision
import torch.nn.functional as nnf
from PIL import Image

import requests

torch.manual_seed(3)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open("C:\\Users\\Simon\\Desktop\\000000039769.jpg")

image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224",use_fast=False)
model = DeiTForImageClassificationWithTeacher.from_pretrained("facebook/deit-base-distilled-patch16-224")

inputs = image_processor(images=image, return_tensors="pt")

outputs = model(**inputs)
logits = outputs.logits

# model predicts one of the 1000 ImageNet classes
prob = nnf.softmax(logits, dim=1)
k = 5
top_p, top_class = prob.topk(k, dim = 1)
predicted_class_idx = logits.argmax(-1).item()

probs = top_p.tolist()
classes = top_class.tolist()

print("Predicted class:", model.config.id2label[predicted_class_idx])
for i in range(k):
    label = model.config.id2label[classes[0][i]]
    prob = probs[0][i]
    print(label,prob)