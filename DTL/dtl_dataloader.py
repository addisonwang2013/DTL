#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import json

from transform_event import *
from collections import namedtuple
import glob


# enable eager mode

class EventAPS_Dataset(Dataset):
    def __init__(self, cfg, mode='train', crop_size=(256, 512), *args, **kwargs):
        super(EventAPS_Dataset, self).__init__(*args, **kwargs)
        assert mode in ('ldr', 'general')

        self.mode = mode
        self.cfg = cfg
        self.crop_h, self.crop_w = crop_size[0], crop_size[1]
        self.imgs = {}
        imgnames = []
        impth = osp.join('./dataset/dtl_data', 'images', mode)

        images = glob.glob(osp.join(impth, '*.png'))
        names = [osp.basename(el.split('.')[1]) for el in images]
        impths = images
        imgnames.extend(names)
        self.imnames = imgnames
        self.len = len(self.imnames)

        self.imgs.update(dict(zip(names, impths)))


        ## pre-processing
        self.to_tensor_event = transforms.Compose([
            transforms.ToTensor(),
        ])


    def __getitem__(self, idx):

        fn = self.imnames[idx]
        impth = self.imgs[fn]
        img = Image.open(impth).convert("RGB")

        name = os.path.splitext(os.path.basename(impth))[0]
        w, h = img.size
        w2 = int(w / 2)

        event = img.crop((0, 0, w2, h))  # crop entire image to get event
        # apply the same transform to both A and B
        event = event.resize((self.crop_w, self.crop_h), Image.BICUBIC)  # resize event
        event = self.to_tensor_event(event)

        return event, name

    def __len__(self):
        return self.len

    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label


if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    ds = EventAPS_Dataset('./dataset/eventdataset', mode='train')
    dl = DataLoader(ds,
                    batch_size=4,
                    shuffle=True,
                    num_workers=4,
                    drop_last=True)
    for imgs, label in dl:
        print(len(imgs))
        for el in imgs:
            print(el.size())
        break
