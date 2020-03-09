import os
import numpy as np
import random
import torch
import copy
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
        self.mask = np.load('mask_8.npy')

    def __len__(self):
        return len(self.IMG_list)
    
    def flip(self, img):
        if self.mode=='train':
            if random.random() < 0.5:
                img = np.fliplr(img)
        return img

    def normalize(self, img):
        img_zero = img - np.amin(img)
        img_one = img_zero / (np.amax(img_zero)+1e-10)
        # img_one_one = img_one * 2.0 - 1.0
        return img_one

    def image_in_painting(self, img):
        x = copy.deepcopy(img)
        img_rows, img_cols = x.shape
        cnt = 5
        while cnt > 0 and random.random() < 0.95:
            block_noise_size_x = random.randint(img_rows//6, img_rows//3)
            block_noise_size_y = random.randint(img_cols//6, img_cols//3)
            noise_x = random.randint(3, img_rows-block_noise_size_x-3)
            noise_y = random.randint(3, img_cols-block_noise_size_y-3)
            x[
            noise_x:noise_x+block_noise_size_x, 
            noise_y:noise_y+block_noise_size_y] = np.random.rand(block_noise_size_x, 
                                                                block_noise_size_y) * 1.0
            cnt -= 1
        return x

    def __getitem__(self, idx, center_frac=0.04, R=8, sparse_recon=False):
        target = Image.open(os.path.join(self.folder, 
                                        self.mode, self.IMG_list[idx]))
        target_np = np.array(target)
        if target_np.ndim > 2:
            print('[!!!] PIL read image shape:', target_np.shape)
            target_np = target_np[:,:,0]
        target_np = self.flip(target_np)
        # print('Target image shape: ', target_np.shape)
        target_np = self.normalize(target_np)
        if sparse_recon:
            fshift = np.fft.fftshift(np.fft.fft2(target_np))
            fshift = fshift * self.mask
            f = np.fft.ifftshift(fshift)
            input_np = self.normalize(np.real(np.fft.ifft2(f)))
        else:
            input_np = self.image_in_painting(target_np)
            
        input_np = np.moveaxis(np.tile(input_np[:,:,None], [1,1,3]), -1, 0)
        target_np = np.moveaxis(np.tile(target_np[:,:,None], [1,1,3]), -1, 0)

        sample = {'inputs': input_np, 'targets': target_np}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class ToTensor(object):
    def __call__(self, sample):
        input, target = sample['inputs'], sample['targets']
        return {'inputs': torch.from_numpy(input),
                'targets': torch.from_numpy(target)}
