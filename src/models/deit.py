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
from src.models.model_base import ModelBase


class Deit(ModelBase):
    def __init__(self,device, num_classes, path = None, pretrained=True):
        super().__init__(
            "deit",
            "facebook/deit-tiny-distilled-patch16-224",
            "timm/deit_tiny_patch16_224.fb_in1k",
            device,
            num_classes,
            path,
            pretrained
        )