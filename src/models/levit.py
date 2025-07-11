import timm
import torch
from PIL import Image
from transformers import LevitImageProcessor
from torcheval.metrics import MulticlassF1Score, MulticlassRecall, MulticlassPrecision, MulticlassAccuracy
from torchvision import datasets, transforms
from safetensors.torch import save_file, load_file
from torch.utils.flop_counter import FlopCounterMode

from src.models.model_base import ModelBase
from src.other.hooks import HookingManager
from src.other.stats import Stats

class LeVit(ModelBase):
    def __init__(self,device, num_classes : int, path : str = None, pretrained=True):
        super().__init__(
            "levit",
            "facebook/levit-128S",
            "timm/levit_128s.fb_dist_in1k",
            device,
            num_classes,
            path,
            pretrained
        )