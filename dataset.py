import os
import PIL.Image
import numpy as np
import torch
from shutil import copyfile
from torchvision import transforms
from torch.utils.data import Dataset


class BayerToTensor(object):
    def __init__(self, scale=1023):
        super(BayerToTensor, self).__init__()
        self.scale = scale


    def __call__(self, sample):
        sample = torch.FloatTensor(np.asarray(sample)[np.newaxis,...])
        return torch.stack([sample[0, 1::2, 1::2], sample[0, 0::2, 1::2], sample[0, 0::2, 0::2], sample[0, 1::2, 0::2]], 0) / self.scale


class DatasetZRR(Dataset):
    def __init__(self, sourcedir, split='train', raw_transforms=[], rgb_transforms=[], refined=True):
        super(DatasetZRR, self).__init__()
        assert split in ['train', 'test']
        if split=='test':
            refined = False
        if refined and not os.path.exists(os.path.join(sourcedir, 'refined')):
            self.refine(sourcedir)
        self.sourcedir = os.path.join(sourcedir, 'refined', split) if refined else os.path.join(sourcedir, split)
        self.rawdir = os.path.join(self.sourcedir, 'huawei_raw')
        self.rgbdir = os.path.join(self.sourcedir, 'canon')
        self.shape = (448, 448)

        self.set_raw_transforms(raw_transforms)
        self.set_rgb_transforms(rgb_transforms)


    def __len__(self):
        return len(os.listdir(self.rawdir))


    def __getitem__(self, idx):
        raw_image = self.raw_transforms(PIL.Image.open(os.path.join(self.rawdir, f'{idx}.png')))
        rgb_image = self.rgb_transforms(PIL.Image.open(os.path.join(self.rgbdir, f'{idx}.jpg')))
        return {'raw':raw_image, 'rgb':rgb_image}


    def set_raw_transforms(self, raw_transforms):
        self.raw_transforms = transforms.Compose(raw_transforms+[BayerToTensor()])


    def set_rgb_transforms(self, rgb_transforms):
        self.rgb_transforms = transforms.Compose(rgb_transforms+[transforms.ToTensor()])


    def refine(self, sourcedir):
        os.makedirs(os.path.join(sourcedir, 'refined', 'train'))
        os.makedirs(os.path.join(sourcedir, 'refined', 'train', 'huawei_raw'))
        os.makedirs(os.path.join(sourcedir, 'refined', 'train', 'canon'))

        sample_idx = 0
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils', 'filter_imageid.txt'), 'r') as f:
            filter_imageid = f.read().splitlines()
        imageid = [id.split('.')[0] for id in os.listdir(os.path.join(sourcedir, 'train', 'huawei_raw'))]
        imageid.sort()
        for id in imageid:
            if id in filter_imageid:
                print(f'skipping imageid {id}')
                continue
            copyfile(os.path.join(sourcedir, 'train', 'huawei_raw', f'{id}.png), os.path.join(sourcedir, 'refined', 'train', 'huawei_raw', f'{sample_idx}.png'))
            copyfile(os.path.join(sourcedir, 'train', 'canon', f'{id}.png'), os.path.join(sourcedir, 'refined', 'train', 'canon', f'{sample_idx}.png'))
            sample_idx += 1
