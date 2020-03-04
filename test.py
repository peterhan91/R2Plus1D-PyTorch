import torch
import os
from eval import make_pred_multilabel
from network import R2Plus1DClassifier

MR_dir = '/media/tianyu.han/mri-scratch/DeepLearning/MRKnee/MRNet-v1.0/'
path='mrnet.pth.tar'
num_classes = 3
layer_sizes = [3, 4, 6, 3]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)
model = R2Plus1DClassifier(num_classes=num_classes, layer_sizes=layer_sizes).to(device)
if os.path.exists(path):
    checkpoint = torch.load(path)
    print("Reloading from previously saved checkpoint")
    model.load_state_dict(checkpoint['state_dict'])

make_pred_multilabel(model, MR_dir)