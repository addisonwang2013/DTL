#!/usr/bin/python
# -*- encoding: utf-8 -*-


#


'''for aps and event fusion--------20210610--------------------'''


class Config(object):  ### settings for 0912 for training_6channels_events
    def __init__(self):
        ## model and loss_folder
        self.ignore_label = 255
        self.aspp_global_feature = False
        ## dataset
        self.n_classes = 6
        self.datapth = './data/'
        self.n_workers = 16
        self.crop_size_event = (256, 256)
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        self.respth = './res'
        self.save_path = './ckpt/MVSEC_depth_seg_20211014/generalloss'
        self.save_dir = './outputs/20211127_test/'

        ## eval control
        self.eval_batchsize = 1
        self.eval_n_workers = 2
        self.eval_scales = [1]
        self.eval_flip = False


