import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
import time

sys.path.insert(0, '../../src/models')

from deit import Deit
from effnet import EffNet
from levit import LeVit


torch.manual_seed(3)

# Configuration
device_str = "cuda" if torch.cuda.is_available() else "cpu"
print("running on",device_str)
device = torch.device(device_str)
batch_size = 16
average = "weighted" # https://docs.pytorch.org/torcheval/stable/generated/torcheval.metrics.functional.multiclass_f1_score.html#torcheval.metrics.functional.multiclass_f1_score
model_selection = "levit"

def test(d):
    d = torch.device(d)
    models = {
        "deit" : Deit(d, 525, "../../networks/birds_deit/weights_final.safetensors"),
        "effnet" : EffNet(d, 525, "../../networks/birds_effnet/weights_final.safetensors"),
        "levit" : LeVit(d, 525, "../../networks/birds_levit/weights_final.safetensors")
    }
    model = models[model_selection]

    test_image = "C:\\Users\\Simon\\Desktop\\blaumeise_1.jpg"

    print('start warmup')
    for i in range(100):
        model.infer(test_image)

    print('start test')
    results = []
    for i in range(100):
        start = time.time()
        model.infer(test_image)
        stop = time.time()
        duration = stop-start
        results.append(duration)
        print(f'[{i}] [{d}]: {duration}')

    return results

cpu_res = test("cpu")
gpu_res = test(device_str)

path = f"../../networks/birds_{model_selection}/inferenceTime.csv"

with open(f"{path}","w") as f:
    f.write(f"i;cpu;gpu\n")
    for i in range(100):
        f.write(f"{i};{cpu_res[i]};{gpu_res[i]}\n")