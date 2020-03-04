import os
import numpy as np
import random
import torch
from PIL import Image
from skimage import exposure
from skimage import util
from torch.utils.data import Dataset
# import FastMRI.transforms as T
# from FastMRI.subsampleK import RandomMaskFunc

class FASTMRIDataset(Dataset):

    def __init__(self, folder,
                mode = 'train',
                transform=None):
        self.mode = mode
        self.transform = transform
        self.folder = folder
        self.IMG_list = os.listdir(os.path.join(self.folder, self.mode))
        self.mask = np.load('fastmri_mask4.npy')

    def __len__(self):
        return len(self.IMG_list)
    
    def flip(self, img):
        if self.mode=='train':
            if random.random() < 0.5:
                img = np.fliplr(img)
        return img

    def normalize(self, img):
        return (img - np.amin(img)) / np.amax(img - np.amin(img))

    def __getitem__(self, idx, center_frac=0.04, R=8):
        target = Image.open(os.path.join(self.folder, 
                                        self.mode, self.IMG_list[idx]))
        target_np = np.array(target)
        if target_np.ndim > 2:
            print('[!!!] PIL read image shape:', target_np.shape)
            target_np = target_np[:,:,0]
        target_np = self.flip(np.flipud(target_np))
        # print('Target image shape: ', target_np.shape)
        target_np = self.normalize(target_np)
        fshift = np.fft.fftshift(np.fft.fft2(target_np))
        fshift = fshift * self.mask
        f = np.fft.ifftshift(fshift)
        input_np = self.normalize(np.real(np.fft.ifft2(f)))
        sample = {'inputs': np.expand_dims(input_np, axis=0), 
                    'targets': np.expand_dims(target_np, axis=0)}
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class ToTensor(object):
    def __call__(self, sample):
        input, target = sample['inputs'], sample['targets']
        return {'inputs': torch.from_numpy(input),
                'targets': torch.from_numpy(target)}
