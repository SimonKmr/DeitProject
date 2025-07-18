import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
import torch
from PIL import Image
import torch.utils.benchmark as pbenchmark
import os
sys.path.insert(0, '../../src/models')

from deit import Deit
from effnet import EffNet
from levit import LeVit

torch.manual_seed(3)

# Configuration
device_str = "cuda" if torch.cuda.is_available() else "cpu"
print("running on",device_str)
device = torch.device(device_str)
batch_size = 8
average = "weighted" # https://docs.pytorch.org/torcheval/stable/generated/torcheval.metrics.functional.multiclass_f1_score.html#torcheval.metrics.functional.multiclass_f1_score
sel_test = ""


def benchmark(m, x):
    m.model(x)

def run(model,n):
    test_tensor = (n, 3, 224, 224)
    test_tensor = torch.randn(test_tensor).to(model.device)
    model.model.eval()
    t0 = pbenchmark.Timer(
        stmt = 'benchmark(m,x)',
        setup = 'from __main__ import benchmark',
        globals={'m':model,'x':test_tensor}
    )
    res = t0.blocked_autorange(min_run_time=60)

    return res

def test(model, n, path_weights, path_inf_log):

    if not os.path.exists(path_inf_log):
        os.makedirs(path_inf_log)

    model_cpu = model('cpu', 525, path_weights)
    res_model_cpu = run(model_cpu, n)

    model_gpu = model('cuda', 525, path_weights)
    res_model_gpu = run(model_gpu, n)

    with open(f"{path_inf_log}/n{n}.csv", "w") as f:
        f.write(f'iteration;gpu;cpu\n')
        n = min(len(res_model_gpu.raw_times), len(res_model_cpu.raw_times))
        for i in range(n):
            f.write(f'{i};{res_model_gpu.raw_times[i]};{res_model_cpu.raw_times[i]}\n')

print('testing deit')
test(Deit, 64, "../../networks/birds_deit/weights/weights_final.safetensors", "../../networks/birds_deit/inference_time")
print('testing levit')
test(LeVit, 64,"../../networks/birds_levit/weights/weights_final.safetensors", "../../networks/birds_levit/inference_time")
print('testing effnet')
test(EffNet, 64,"../../networks/birds_effnet/weights/weights_final.safetensors", "../../networks/birds_effnet/inference_time")

quit()

for i in range(1,17):
    print('testing deit')
    test(Deit,i, "../../networks/birds_deit/weights/weights_final.safetensors", "../../networks/birds_deit/inference_time")
    print('testing levit')
    test(LeVit, i,"../../networks/birds_levit/weights/weights_final.safetensors", "../../networks/birds_levit/inference_time")
    print('testing effnet')
    test(EffNet, i,"../../networks/birds_effnet/weights/weights_final.safetensors", "../../networks/birds_effnet/inference_time")
