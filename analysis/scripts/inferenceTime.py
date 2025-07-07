import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
import torch
from PIL import Image
import torch.utils.benchmark as pbenchmark

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

def benchmark(m, x):
    m.model.eval()
    m.model(x)

def run(model,n):
    test_tensor = (n, 3, 224, 224)
    test_tensor = torch.randn(test_tensor).to(model.device)
    t0 = pbenchmark.Timer(
        stmt = 'benchmark(m,x)',
        setup = 'from __main__ import benchmark',
        globals={'m':model,'x':test_tensor}
    )
    res = t0.blocked_autorange(min_run_time=600)

    return res

test_image_path = "C:\\Users\\Simon\\Desktop\\blaumeise_1.jpg"
test_image = Image.open(test_image_path).convert('RGB')

print('start test')
print('testing effnet')
n = 1
effnet_gpu = EffNet('cuda', 525, "../../networks/birds_effnet/weights_final.safetensors")
res_effnet_gpu = run(effnet_gpu,n)

effnet_cpu = EffNet('cpu', 525, "../../networks/birds_effnet/weights_final.safetensors")
res_effnet_cpu = run(effnet_cpu,n)

print('testing levit')
levit_gpu = LeVit('cuda', 525, "../../networks/birds_levit/weights_final.safetensors")
res_levit_gpu = run(levit_gpu,n)

levit_cpu = LeVit('cpu', 525, "../../networks/birds_levit/weights_final.safetensors")
res_levit_cpu = run(levit_cpu,n)

print('testing deit')
deit_gpu = Deit('cuda', 525, "../../networks/birds_deit/weights_final.safetensors")
res_deit_gpu = run(deit_gpu,n)

deit_cpu = Deit('cpu', 525, "../../networks/birds_deit/weights_final.safetensors")
res_deit_cpu = run(deit_cpu,n)

print(res_effnet_gpu)
print(res_effnet_cpu)
print(res_levit_gpu)
print(res_levit_cpu)
print(res_deit_gpu)
print(res_deit_cpu)

res_effnet_gpu_mean = res_effnet_gpu.mean
res_levit_gpu_mean = res_levit_gpu.mean
res_deit_gpu_mean = res_deit_gpu.mean

print(f"deit: {res_deit_gpu_mean}, levit: {res_levit_gpu_mean}, effnet: {res_effnet_gpu_mean}")

res_effnet_cpu_mean = res_effnet_cpu.mean
res_levit_cpu_mean = res_levit_cpu.mean
res_deit_cpu_mean = res_deit_cpu.mean

print(f"deit: {res_deit_cpu_mean}, levit: {res_levit_cpu_mean}, effnet: {res_effnet_cpu_mean}")
with open(f"../../networks/birds_deit/inferenceTime.csv", "w") as f:
    f.write(f'iteration;gpu;cpu\n')
    n = min(len(res_deit_gpu.raw_times),len(res_deit_cpu.raw_times))
    for i in range(n):
        f.write(f'{i};{res_deit_gpu.raw_times[i]};{res_deit_cpu.raw_times[i]}\n')

with open(f"../../networks/birds_levit/inferenceTime.csv", "w") as f:
    f.write(f'iteration;gpu;cpu\n')
    n = min(len(res_levit_gpu.raw_times),len(res_levit_cpu.raw_times))
    for i in range(n):
        f.write(f'{i};{res_levit_gpu.raw_times[i]};{res_levit_cpu.raw_times[i]}\n')

with open(f"../../networks/birds_effnet/inferenceTime.csv", "w") as f:
    f.write(f'iteration;gpu;cpu\n')
    n = min(len(res_effnet_gpu.raw_times),len(res_effnet_cpu.raw_times))
    for i in range(n):
        f.write(f'{i};{res_effnet_gpu.raw_times[i]};{res_effnet_cpu.raw_times[i]}\n')