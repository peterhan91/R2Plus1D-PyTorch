import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import copy
import sys
sys.path.insert(1, '/media/tianyu.han/mri-scratch/DeepLearning/MRKnee/Script/R2Plus1D-PyTorch/')
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchsummary import summary
from torch.autograd import Variable
# from dataset import MRIDataset, ToTensor
from torchvision import transforms, utils
from inflate_src.i3res import I3ResNet
from ResUnet.RESUNET import ResNetUNet

from integrated_grad import IntegratedGradients
from guided_grad import GuidedBackprop 
from integrated_guided_grad import IntegratedGuidedGradients

def crop(buffer, crop_size=224):
    # buffer shape: [16, 256, 256]
    height_index = (buffer.shape[1] - crop_size) // 2
    width_index = (buffer.shape[2] - crop_size) // 2
    return buffer[:, height_index:height_index + crop_size,
                    width_index:width_index + crop_size]

def normalize_frame(frame):
    frame_zero = frame - np.amin(frame)
    frame_one = frame_zero / (np.amax(frame_zero)+1e-10)
    return frame_one

def loadscans(scanname, view_ID):
    clip_len = 32
    resize_height = 256
    resize_width = 256
    
    scan = np.load(scanname)
    if scan.ndim != 3:
        print('Not 3D input scan numpy array!!!, it has a shape of', scan.shape)
    slice_count = int(scan.shape[0])
    slice_width = int(scan.shape[1])
    slice_height = int(scan.shape[2])
    buffer = np.empty((clip_len, resize_height, resize_width), np.dtype('float32'))
    if clip_len < slice_count:
        seq = list(range(slice_count))
        seq.sort(key=int)
    else:
        seq = list(range(slice_count))
        while len(seq) < clip_len:
            seq.insert(int(slice_count/2), int(slice_count/2))
    count = 0
    while count < clip_len:
        index = seq[count]
        frame = scan[index]
        if (slice_height != resize_height) or (slice_width != resize_width):
            frame = cv2.resize(frame, (resize_width, resize_height))
        frame = normalize_frame(frame)
        # frame -= self.mean[view_ID]
        # frame /= self.std[view_ID]
        buffer[count] = frame # normalize all slices to the range of [0, 1]
        count += 1
    return buffer 

def prepare_input(folder, filename, views=['sagittal', 'coronal', 'axial'], rijeka=True):
    buffers = []
    for view_ID, view in enumerate(views):
        if rijeka:
            name = os.path.join(folder, filename)
        else:
            name = os.path.join(folder, view, filename)
        buffer = loadscans(name, view_ID) # shape [32, 256, 256]
        # print(buffer.shape) # debug
        buffer = crop(buffer) # shape [16, 224, 224]
        buffers.append(buffer)
    return torch.from_numpy(np.expand_dims(np.array(buffers), axis=0))

# path = './OLD_model/ResNet/resnet34_ImageNet/'+'mrnet.pth.tar'
# folder = '/media/tianyu.han/mri-scratch/DeepLearning/MRKnee/MRNet-v1.0/valid/'
# filename = '1200.npy'
path = './OLD_model/rijeka/resnet34_finetune_ImageNet/'+'mrnet.pth.tar'
folder = '/media/tianyu.han/mri-scratch/DeepLearning/MRKnee/Script/Rijeka/test/'
filename = '498724-5.npy'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)
resnet = torchvision.models.resnet34(pretrained=True)
model = I3ResNet(copy.deepcopy(resnet), 3).to(device)
if os.path.exists(path):
    checkpoint = torch.load(path)
    print("Reloading from previously saved checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
model = model.eval()
inputs = prepare_input(folder, filename)
prep_input = Variable(inputs.cuda(), requires_grad=True)
# IG = IntegratedGradients(model)
GB = GuidedBackprop(model)
# grads = IG.generate_integrated_gradients(prep_input, [0], 100)
grads = GB.generate_gradients(prep_input, [0])

save_name = 'GB'+ filename
np.save(save_name, grads)
save_name = 'input_'+filename
np.save(save_name, inputs[0])


