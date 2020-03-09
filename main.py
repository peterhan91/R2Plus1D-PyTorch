from trainer import train_model
from test import test_model
import tensorlayer as tl
import torchvision
from ResUnet.RESUNET import ResNetUNet
import os
import torch

vision_model = False

MR_dir = '/media/tianyu.han/mri-scratch/DeepLearning/MRKnee/MRNet-v1.0/'
out_dir = './OLD_model/resnet50_inpaint/'
res_path = './ResUnet/old_model/MRNet/Res50_inpaint/saved.pth.tar'
tl.files.exists_or_mkdir(out_dir)
if vision_model:
    resnet = torchvision.models.resnet50(pretrained=False)
else:
    resunet = ResNetUNet()
    if os.path.exists(res_path):
        checkpoint = torch.load(res_path)
        print("Reloading ResNet from previously saved checkpoint")
        resunet.load_state_dict(checkpoint['state_dict'])
    resnet = resunet.base_model
train_model(resnet, 3, MR_dir, path=out_dir+'mrnet.pth.tar')
test_model(resnet, out_dir)