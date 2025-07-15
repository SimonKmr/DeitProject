import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
import torch
from PIL import Image
import torch.utils.benchmark as pbenchmark
from torchvision import datasets
from torch.utils.data import DataLoader

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

def length():
    valid_folder = "D:\\Datasets\\bird-species-dataset\\data\\valid"
    valid_dataset = datasets.ImageFolder(root=valid_folder)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    return len(valid_loader) * batch_size

print(f"number of pictures: {length()}")

def benchmark(m, x):
    m.model.eval()
    with torch.no_grad():
        for data in x:
            inputs, _ = data
            inputs = inputs.to(m.device)
            _ = m.model(inputs)

def run(model):
    # Load validation set
    valid_folder = "D:\\Datasets\\bird-species-dataset\\data\\valid"
    valid_dataset = datasets.ImageFolder(root=valid_folder, transform=model.transform)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    t0 = pbenchmark.Timer(
        stmt = 'benchmark(m,x)',
        setup = 'from __main__ import benchmark',
        globals={'m':model,'x':valid_loader}
    )
    res = t0.blocked_autorange(min_run_time=600)

    return res

test_image_path = "C:\\Users\\Simon\\Desktop\\blaumeise_1.jpg"
test_image = Image.open(test_image_path).convert('RGB')

print('start test')
print('testing effnet')
effnet_gpu = EffNet('cuda', 525, "../../networks/birds_effnet/weights/weights_final.safetensors")
res_effnet_gpu = run(effnet_gpu)

effnet_cpu = EffNet('cpu', 525, "../../networks/birds_effnet/weights/weights_final.safetensors")
res_effnet_cpu = run(effnet_cpu)

res_effnet_gpu_mean = res_effnet_gpu.mean
res_effnet_cpu_mean = res_effnet_cpu.mean

print(res_effnet_gpu)
print(res_effnet_cpu)

with open(f"../../networks/birds_effnet/throughput.csv", "w") as f:
    f.write(f'iteration;gpu;cpu\n')
    n = min(len(res_effnet_gpu.raw_times),len(res_effnet_cpu.raw_times))
    for i in range(n):
        f.write(f'{i};{res_effnet_gpu.raw_times[i]};{res_effnet_cpu.raw_times[i]}\n')

print('testing levit')
levit_gpu = LeVit('cuda', 525, "../../networks/birds_levit/weights/weights_final.safetensors")
res_levit_gpu = run(levit_gpu)

levit_cpu = LeVit('cpu', 525, "../../networks/birds_levit/weights/weights_final.safetensors")
res_levit_cpu = run(levit_cpu)

res_levit_gpu_mean = res_levit_gpu.mean
res_levit_cpu_mean = res_levit_cpu.mean

print(res_levit_gpu)
print(res_levit_cpu)

with open(f"../../networks/birds_levit/throughput.csv", "w") as f:
    f.write(f'iteration;gpu;cpu\n')
    n = min(len(res_levit_gpu.raw_times),len(res_levit_cpu.raw_times))
    for i in range(n):
        f.write(f'{i};{res_levit_gpu.raw_times[i]};{res_levit_cpu.raw_times[i]}\n')

print('testing deit')
deit_gpu = Deit('cuda', 525, "../../networks/birds_deit/weights/weights_final.safetensors")
res_deit_gpu = run(deit_gpu)

deit_cpu = Deit('cpu', 525, "../../networks/birds_deit/weights/weights_final.safetensors")
res_deit_cpu = run(deit_cpu)

res_deit_cpu_mean = res_deit_cpu.mean
res_deit_gpu_mean = res_deit_gpu.mean

print(res_deit_gpu)
print(res_deit_cpu)

with open(f"../../networks/birds_deit/throughput.csv", "w") as f:
    f.write(f'iteration;gpu;cpu\n')
    n = min(len(res_deit_gpu.raw_times),len(res_deit_cpu.raw_times))
    for i in range(n):
        f.write(f'{i};{res_deit_gpu.raw_times[i]};{res_deit_cpu.raw_times[i]}\n')