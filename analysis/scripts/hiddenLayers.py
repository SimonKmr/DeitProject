import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
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
model_selection = "effnet"
models = {
    "deit" : Deit(device, 525, "../../networks/birds_deit/weights_final.safetensors"),
    "effnet" : EffNet(device, 525, "../../networks/birds_effnet/weights_final.safetensors"),
    "levit" : LeVit(device, 525, "../../networks/birds_levit/weights_final.safetensors")
}

model = models[model_selection]
layers = model.infer_hooked("C:\\Users\\Simon\\Desktop\\blaumeise_1.jpg",2)
#print(layers)

print(np.shape(layers))

# transformer
def visualize_attention_map():
    print('processing layer')
    im2 = plt.imshow(layers[0, :, :].cpu().numpy(), cmap=plt.cm.viridis, alpha=.9, interpolation='bilinear')
    plt.colorbar()
    plt.savefig('feature_map/deit_1.png')
    plt.close()

def map_attention():
    return

# convNets
def visualize_convolution_layer():
    dim_i = layers[0, :, :, :].size(dim=0)
    for i in range(dim_i):
        print('processing layer %s' % (i + 1))
        im2 = plt.imshow(layers[0, i, :, :].cpu().numpy(), cmap=plt.cm.viridis, alpha=.9, interpolation='bilinear')
        plt.colorbar()
        plt.savefig('feature_map/cnn_1_%s.png' % (i))
        plt.close()

visualize_convolution_layer()