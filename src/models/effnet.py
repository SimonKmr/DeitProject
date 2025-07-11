import timm
import torch
from PIL import Image
from transformers import EfficientNetImageProcessor
from torcheval.metrics import MulticlassF1Score, MulticlassRecall, MulticlassPrecision, MulticlassAccuracy
from torchvision import datasets, transforms
from safetensors.torch import save_file, load_file

from src.models.model_base import ModelBase
from src.other.hooks import HookingManager
from src.other.stats import Stats
from torch.utils.flop_counter import FlopCounterMode

class EffNet(ModelBase):
    def __init__(self,device, num_classes, path = None, pretrained=True):
        super().__init__(
            "effnet",
            "google/efficientnet-b0",
            "timm/efficientnet_b0.ra_in1k",
            device,
            num_classes,
            path,
            pretrained
        )