import os
import numpy as np
import random
import torch
import copy
from PIL import Image
from skimage import exposure
from skimage import util
from torch.utils.data import Dataset
from skimage.transform import rotate
try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb

class FASTMRIDataset(Dataset):

    def __init__(self, folder,
                mode = 'train',
                sparserecon=False,
                transform=None):
        self.mode = mode
        self.transform = transform
        self.folder = folder
        self.sparserecon = sparserecon
        self.IMG_list = os.listdir(os.path.join(self.folder, self.mode))
        self.mask = np.load('mask_8.npy')

    def __len__(self):
        return len(self.IMG_list)
    
    def generate_mask(self, img, R=8):
        h, w = img.shape
        mask = np.zeros_like(img)
        assert h==w
        if self.mode=='train':
            # non_zeros = int(h/R)
            # for non_zero in random.sample(list(range(h)), non_zeros):
            #     mask[non_zero] = 1.
            height_index = (h - h//R) // 2
            mask[height_index:height_index + h//R,:] = 1.
        else:
            height_index = (h - h//R) // 2
            mask[height_index:height_index + h//R,:] = 1.
        return mask

    def flip(self, img):
        if self.mode=='train':
            if random.random() < 0.5:
                img = np.fliplr(img)
        return img

    def rotation(self, img):
        if self.mode == 'train':
            if random.random() < 0.5:
                img = rotate(img, 90, resize=True)
        return img

    def normalize(self, img):
        img_zero = img - np.amin(img)
        img_one = img_zero / (np.amax(img_zero)+1e-10)
        # img_one_one = img_one * 2.0 - 1.0
        return img_one

    def bezier_curve(self, points, nTimes=1000):
        
        def bernstein_poly(i, n, t):
            return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

        nPoints = len(points)
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])
        t = np.linspace(0.0, 1.0, nTimes)
        polynomial_array = np.array([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])
        xvals = np.dot(xPoints, polynomial_array)
        yvals = np.dot(yPoints, polynomial_array)
        return xvals, yvals

    def nonlinear_transformation(self, x, prob=0.5):
        if random.random() >= prob:
            return x
        points = [[0, 0], [random.random(), random.random()], 
                    [random.random(), random.random()], [1, 1]]
        xpoints = [p[0] for p in points]
        ypoints = [p[1] for p in points]
        xvals, yvals = self.bezier_curve(points, nTimes=100000)
        if random.random() < 0.5:
            # Half change to get flip
            xvals = np.sort(xvals)
        else:
            xvals, yvals = np.sort(xvals), np.sort(yvals)
        nonlinear_x = np.interp(x, xvals, yvals)
        return nonlinear_x

    def local_pixel_shuffling(self, x, prob=0.5):
        if random.random() >= prob:
            return x
        image_temp = copy.deepcopy(x)
        orig_image = copy.deepcopy(x)
        img_rows, img_cols = x.shape
        num_block = 10000
        for _ in range(num_block):
            block_noise_size_x = random.randint(1, img_rows//10)
            block_noise_size_y = random.randint(1, img_cols//10)
            noise_x = random.randint(0, img_rows-block_noise_size_x)
            noise_y = random.randint(0, img_cols-block_noise_size_y)
            window = orig_image[noise_x:noise_x+block_noise_size_x, 
                                noise_y:noise_y+block_noise_size_y]
            window = window.flatten()
            np.random.shuffle(window)
            window = window.reshape((block_noise_size_x, 
                                    block_noise_size_y))
            image_temp[noise_x:noise_x+block_noise_size_x, 
                        noise_y:noise_y+block_noise_size_y] = window
        return image_temp    
    
    def image_in_painting(self, img):
        x = copy.deepcopy(img)
        img_rows, img_cols = x.shape
        cnt = 5
        while cnt > 0 and random.random() < 0.95:
            block_noise_size_x = random.randint(img_rows//6, img_rows//3)
            block_noise_size_y = random.randint(img_cols//6, img_cols//3)
            noise_x = random.randint(3, img_rows-block_noise_size_x-3)
            noise_y = random.randint(3, img_cols-block_noise_size_y-3)
            x[noise_x:noise_x+block_noise_size_x, 
            noise_y:noise_y+block_noise_size_y] = np.random.rand(block_noise_size_x, 
                                                                block_noise_size_y) * 1.0
            cnt -= 1
        return x

    def center_crop(self, img, crop_size = 128):
        height_index = (img.shape[0] - crop_size) // 2
        width_index = (img.shape[1] - crop_size) // 2
        img = img[height_index:height_index + crop_size,
                        width_index:width_index + crop_size]
        return img

    def crop(self, img, crop_size = 128):
        # buffer shape: [16, 256, 256]
        if self.mode=='train':
            height_index = np.random.randint(img.shape[0] - crop_size)
            width_index = np.random.randint(img.shape[1] - crop_size)
        else:
            height_index = (img.shape[0] - crop_size) // 2
            width_index = (img.shape[1] - crop_size) // 2

        img = img[height_index:height_index + crop_size,
                        width_index:width_index + crop_size]
        return img

    def __getitem__(self, idx):
        target = Image.open(os.path.join(self.folder, 
                                        self.mode, self.IMG_list[idx]))
        target_np = np.array(target)
        if target_np.ndim > 2:
            print('[!!!] PIL read image shape:', target_np.shape)
            target_np = target_np[:,:,0]
        # target_np = self.crop(self.flip(target_np))
        # target_np = self.flip(target_np)
        # print('Target image shape: ', target_np.shape)
        # target_np = self.normalize(target_np)
        if self.sparserecon:
            # target_np = self.normalize(self.flip(target_np))
            target_np = self.rotation(self.normalize(self.crop(self.flip(target_np))))
            fshift = np.fft.fftshift(np.fft.fft2(target_np))
            mask = self.generate_mask(target_np)
            # fshift = fshift * self.center_crop(self.mask)
            fshift = fshift * mask
            f = np.fft.ifftshift(fshift)
            input_np = self.normalize(np.real(np.fft.ifft2(f)))
        else:
            target_np = self.rotation(self.normalize(self.crop(self.flip(target_np))))
            input_np = self.local_pixel_shuffling(target_np, prob=1.5)
            # x = self.nonlinear_transformation(x, prob=0.9)
            # input_np = self.image_in_painting(target_np)
            
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
