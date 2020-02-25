import os
from pathlib import Path

import cv2
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset


class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list 
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            directory (str): The path to the directory containing the train/val/test datasets
            mode (str, optional): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            if the mode==train: random crop, flip, and slice selection are performed; 
            if the mode==val/test: center crop and sequencial slice selection are performed.
            clip_len (int, optional): Determines how many frames are there in each clip. Defaults to 16. 
        """

    def __init__(self, directory, mode='train', clip_len=16):
        self.folder = Path(directory)/mode  # get the directory of the specified split
        self.views = ['sagittal', 'coronal', 'axial']
        self.scanlists = os.listdir(os.path.join(self.folder, self.views[0]))

        self.clip_len = clip_len

        # the following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 256
        self.resize_width = 256
        self.crop_size = 224

        self.df = pd.read_csv('MRnet_labels.csv')
        self.df = self.df[self.df['mode'] == mode]
        self.df = self.df.set_index("Scan Index")
        self.PRED_LABEL = ['abnormality', 'ACL tear', 'meniscal tear'] 

    def __getitem__(self, index):
        # loading and preprocessing. TODO move them to transform classes
        buffer = self.loadvideo(self.fnames[index])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        buffer = self.normalize(buffer)

        return buffer, self.label_array[index]    
        
    def loadscans(self, scanname, randomslices=True):
        scan = np.load(scanname)
        if scan.ndim != 3:
            print('Not 3D input scan numpy array!!!, it has a shape of', scan.shape)
            break
        slice_count = int(scan.shape[0])
        slice_width = int(scan.shape[1])
        slice_height = int(scan.shape[2])
        buffer = np.empty((self.clip_len, self.resize_height, self.resize_width), np.dtype('float32'))
        
        if self.clip_len < slice_count:
            if randomslices:
                seq = random.sample(range(slice_count), self.clip_len)
            else:
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
            buffer[count] = frame
            count += 1

        return buffer            

    def normalize(self, buffer):
        # Normalize the buffer
        # NOTE: Default values of RGB images normalization are used, as precomputed 
        # mean and std_dev values (akin to ImageNet) were unavailable for Kinetics. Feel 
        # free to push to and edit this section to replace them if found. 
        buffer = (buffer - 128)/128
        return buffer

    def __len__(self):
        return len(self.fnames)

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}

class VideoDataset1M(VideoDataset):
    r"""Dataset that implements VideoDataset, and produces exactly 1M augmented
    training samples every epoch.
        
        Args:
            directory (str): The path to the directory containing the train/val/test datasets
            mode (str, optional): Determines which folder of the directory the dataset will read from. Defaults to 'train'. 
            clip_len (int, optional): Determines how many frames are there in each clip. Defaults to 8. 
        """
    def __init__(self, directory, mode='train', clip_len=8):
        # Initialize instance of original dataset class
        super(VideoDataset1M, self).__init__(directory, mode, clip_len)

    def __getitem__(self, index):
        # if we are to have 1M samples on every pass, we need to shuffle
        # the index to a number in the original range, or else we'll get an 
        # index error. This is a legitimate operation, as even with the same 
        # index being used multiple times, it'll be randomly cropped, and
        # be temporally jitterred differently on each pass, properly
        # augmenting the data. 
        index = np.random.randint(len(self.fnames))

        buffer = self.loadvideo(self.fnames[index])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        buffer = self.normalize(buffer)

        return buffer, self.label_array[index]    

    def __len__(self):
        return 1000000  # manually set the length to 1 million