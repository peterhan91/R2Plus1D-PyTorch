from trainer import train_model
from test import test_model
import tensorlayer as tl
import torchvision
from ResUnet.RESUNET import ResNetUNet
import os
import torch

vision_model = False
rijeka = True

if rijeka:
    MR_dir = '/media/tianyu.han/mri-scratch/DeepLearning/MRKnee/Script/Rijeka/'
else:
    MR_dir = '/media/tianyu.han/mri-scratch/DeepLearning/MRKnee/MRNet-v1.0/'
out_dir = './OLD_model/rijeka/resnet34_self/'
res_path = './ResUnet/old_model/ModelGenesis/Res34_Pure/saved.pth.tar'
tl.files.exists_or_mkdir(out_dir)
if vision_model:
    resnet = torchvision.models.resnet34(pretrained=True)
    print("Using ResNet from torchvison")
else:
    resunet = ResNetUNet()
    if os.path.exists(res_path):
        checkpoint = torch.load(res_path)
        print("Reloading ResUUUNet from previously saved checkpoint")
        resunet.load_state_dict(checkpoint['state_dict'])
    resnet = resunet.base_model

# train_model(resnet, 3, MR_dir, train_Rijeka=rijeka, path=out_dir+'mrnet.pth.tar')
test_model(resnet, MR_dir, out_dir, 3, test_Rijeka=rijeka)