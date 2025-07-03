import torch
import torch.nn as nn

class HookingManager:
    def __init__(self):
        self.result = {}

    def hook(self, model, input, output):
        self.result = output.detach()
        return