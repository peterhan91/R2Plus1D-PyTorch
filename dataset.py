import os
from pathlib import Path
from skimage.transform import rotate

import cv2
import random
import re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


digits = re.compile(r'(\d+)')
def tokenize(filename):
    return tuple(int(token) if match else token
                for token, match in
                ((fragment, digits.search(fragment))
                for fragment in digits.split(filename)))

class MRIDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list 
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            directory (str): The path to the directory containing the train/val/test datasets
            mode (str, optional): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            if the mode==train: random crop, lr flip, and randomlized slice selection are performed; 
            if the mode==val/test: center crop and sequencial slice selection are performed.
            clip_len (int, optional): Determines how many frames are there in each clip. Defaults to 16. 
        returns:
            buffers: numpy array with shape: [3, 16, 224, 224]
            label: numpy array (1D) with shape: [3,]
    """

    def __init__(self, directory, mode='train', clip_len=16, transform=None):
        self.mode = mode
        self.transform = transform
        self.folder = os.path.join(directory, self.mode)  # get the directory of the specified split
        self.views = ['sagittal', 'coronal', 'axial']
        # self.views = ['sagittal', 'sagittal', 'sagittal']
        self.scanlists = os.listdir(os.path.join(self.folder, self.views[0]))
        self.scanlists.sort(key=tokenize)
        self.clip_len = clip_len
        self.resize_height = 256
        self.resize_width = 256
        self.crop_size = 224

        self.df = pd.read_csv('MRnet_labels.csv')
        self.df = self.df[self.df['fold'] == self.mode]
        self.df = self.df.set_index('Scan Index')
        self.PRED_LABEL = ['abnormality', 'ACL tear', 'meniscal tear']    
        
    def __len__(self):
        return len(self.scanlists)
    
    def loadscans(self, scanname):
        scan = np.load(scanname)
        if scan.ndim != 3:
            print('Not 3D input scan numpy array!!!, it has a shape of', scan.shape)
        slice_count = int(scan.shape[0])
        slice_width = int(scan.shape[1])
        slice_height = int(scan.shape[2])
        buffer = np.empty((self.clip_len, self.resize_height, self.resize_width), np.dtype('float32'))
        
        if self.clip_len < slice_count:
            if self.mode == 'train': 
                 seq = random.sample(range(slice_count), self.clip_len)
                 seq.sort(key=int)
            else:
                # seq = list(range(slice_count))
                # seq.sort(key=int)
                
                cc = int(slice_count/2)
                tt = int(self.clip_len/2)
                if self.clip_len%2 == 0:
                    seq = [x+cc for x in range(-tt,tt)]
                else:
                    seq = [x+cc for x in range(-tt,tt+1)]
    
        else:
            seq = list(range(slice_count))
            while len(seq) < self.clip_len:
                seq.insert(int(slice_count/2), int(slice_count/2))
        
        count = 0
        retaining = True
        while (count < self.clip_len and retaining):
            index = seq[count]
            frame = scan[index]
            if (slice_height != self.resize_height) or (slice_width != self.resize_width):
                frame = cv2.resize(frame, (self.resize_width, self.resize_height))
            buffer[count] = self.normalize_frame(frame) # normalize all slices to the range of [0, 1]
            # buffer[count] = frame
            count += 1

        return buffer 
    
    def crop(self, buffer, crop_size):
        # buffer shape: [16, 256, 256]
        if self.mode=='train':
            height_index = np.random.randint(buffer.shape[1] - crop_size)
            width_index = np.random.randint(buffer.shape[2] - crop_size)
        else:
            height_index = (buffer.shape[1] - crop_size) // 2
            width_index = (buffer.shape[2] - crop_size) // 2

        buffer = buffer[:, height_index:height_index + crop_size,
                        width_index:width_index + crop_size]
        return buffer
    
    def flip(self, buffer):
        # buffer shape: [16, 224, 224]
        if self.mode=='train':
            # perform random lr flip
            for n in range(len(buffer)):
                if random.random() < 0.5:
                    buffer[n] = np.fliplr(buffer[n])
        return buffer
    
    def rotation(self, buffer, degree_range=15.0):
        if self.mode=='train':
            for n in range(len(buffer)):
                if random.random() < 0.5:
                    if random.random() < 0.5:
                        buffer[n] = rotate(buffer[n], degree_range)
                    else:
                        buffer[n] = rotate(buffer[n], degree_range*-1.0)
        return buffer
    
    def normalize_frame(self, frame):
        frame_zero = frame - np.amin(frame)
        frame_one = frame_zero / (np.amax(frame_zero)+1e-10)
        # frame_one_one = frame_one * 2.0 - 1.0
        return frame_one
    
    def __getitem__(self, index):
        buffers = []
        for view in self.views:
            name = os.path.join(self.folder, view, self.scanlists[index])
            buffer = self.loadscans(name)
            # print(buffer.shape) # debug
            buffer = self.crop(buffer, self.crop_size) # shape [16, 224, 224]
            buffer = self.flip(buffer)
            buffer = self.rotation(buffer)
            buffers.append(buffer)
        
        labels = np.zeros(len(self.PRED_LABEL), dtype=int) # one-hot like vector
        for i in range(0, len(self.PRED_LABEL)):
            if(self.df[self.PRED_LABEL[i].strip()].loc[self.scanlists[index]].astype('int') > 0):
            # df.series.str.strip: remove leading and traling characters
                labels[i] = self.df[self.PRED_LABEL[i].strip()].loc[self.scanlists[index]].astype('int')
        sample = {'buffers': np.array(buffers), 'labels': labels}
        # print('debug scan name:', self.scanlists[index], 'scan label:', labels)

        if self.transform:
            sample = self.transform(sample)

        return sample 

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        buffers, labels = sample['buffers'], sample['labels']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        return {'buffers': torch.from_numpy(buffers),
                'labels': torch.from_numpy(labels)}
'''
class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace
    
    def __call__(self, sample):
        buffers, labels = sample['buffers'], sample['labels']
        for n in range(buffers.shape[1]):
            buffers[:,n,:,:] = F.normalize(buffers[:,n,:,:], self.mean, self.std, self.inplace)
        return {'buffers': buffers, 'labels': labels}
'''

