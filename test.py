import torch
import torchvision
import copy
import os
from eval import make_pred_multilabel
from network import R2Plus1DClassifier
from inflate_src.i3res import I3ResNet
from ResUnet.RESUNET import ResNetUNet

def test_model(resnet, save_dir):
    MR_dir = '/media/tianyu.han/mri-scratch/DeepLearning/MRKnee/MRNet-v1.0/'
    path=os.path.join(save_dir, 'mrnet.pth.tar')
    num_classes = 3
    layer_sizes = [3, 4, 6, 3]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)
    # model = R2Plus1DClassifier(num_classes=num_classes, layer_sizes=layer_sizes).to(device)
    # resnet = torchvision.models.resnet50(pretrained=False)
    model = I3ResNet(copy.deepcopy(resnet)).to(device)
    if os.path.exists(path):
        checkpoint = torch.load(path)
        print("Reloading from previously saved checkpoint")
        model.load_state_dict(checkpoint['state_dict'])

    make_pred_multilabel(model, MR_dir, save_dir)