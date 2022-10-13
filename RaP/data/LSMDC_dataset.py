from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image
import torch
import numpy as np
import random
import decord
from decord import VideoReader
import json
import os
from data.utils import pre_caption
# from video_dataset.DiDeMo import decoder
import lmdb
import io
import av 
import pickle
from torchvision import transforms
from data.data_utils import (
    ImageResize, ImagePad, image_to_tensor)
    
decord.bridge.set_bridge("torch")

class ImageNorm(object):

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        
    def __call__(self, img):

        if torch.max(img) > 1 and self.mean.max() <= 1:
            img.div_(255.)
        return img.sub_(self.mean).div_(self.std)

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

class lsmdc_dataset(Dataset):
    def __init__(self,video_root ,data_split, num_frm=4, frm_sampling_strategy="rand", max_img_size=384, video_fmt='.avi'):

        self.text = []
        self.vedio_id = []
        self.video_root=video_root
        false_list = ['3061_SNOW_FLOWER_01.30.05.231-01.30.06.393.avi','3083_TITANIC2_00.04.38.426-00.04.39.883.avi',
        '1057_Seven_pounds_00.23.01.000-00.23.05.457.avi','0028_The_Crying_Game_01.44.50.801-01.44.52.392.avi','3085_TRUE_GRIT_01.37.48.943-01.37.50.169.avi',
        '0014_Ist_das_Leben_nicht_schoen_00.01.45.481-00.02.06.641.avi','1053_Harry_Potter_and_the_philosophers_stone_00.48.23.000-00.48.28.589.avi']
        if data_split=='test':
            file = f'meta_data/LSMDC/LSMDC16_challenge_1000_publictect.csv'
        elif data_split=='val' :
            file =f'meta_data/LSMDC/LSMDC16_annos_val.csv'
        elif data_split=='train':
            file =f'meta_data/LSMDC/LSMDC16_annos_training.csv'
        print('file')
        lines = open(file).readlines()
        if data_split=='val':
            lines=lines[:1000]
        for line in lines:
            ls_now = line.strip().split('\t')

            sub_path = ls_now[0].split('.')[0]
            remove = sub_path.split('_')[-1]
            sub_path = sub_path.replace('_' + remove, '/')
            rel_video_fp = sub_path + ls_now[0] + '.avi'
            full_video_fp = os.path.join(f'{self.video_root}', rel_video_fp)
            false_flag  =0 
            for false_item in false_list:
                if false_item in full_video_fp:
                    false_flag=1
                    break
            if false_flag==0:
                self.text.append(ls_now[-1])
                self.vedio_id.append(full_video_fp)

        self.num_frm = num_frm
        self.frm_sampling_strategy = frm_sampling_strategy
        self.max_img_size = max_img_size
        # self.video_root = video_root
        self.video_fmt = video_fmt

        self.img_resize = ImageResize(
            max_img_size,
            "bilinear")
        self.img_pad = ImagePad(
            max_img_size, max_img_size) 

        self.data_split = data_split
        self.img_norm = ImageNorm(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

        self.pretrain_transform_webvid_train= transforms.Compose([
                transforms.RandomResizedCrop(max_img_size,scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0, saturation=0, hue=0),
                self.img_norm,
            ])
        self.txt2video = [i for i in range(len(self.text))]
        self.video2txt = self.txt2video
            

    def __len__(self):
        return len(self.text)


    def __getitem__(self, index):

        ann = self.text[index]

        vedio_id = self.vedio_id[index]


        video_path = vedio_id

        vid_frm_array = self._load_video_from_path_decord(video_path, height=self.max_img_size, width=self.max_img_size)


        resized_img = self.img_resize(vid_frm_array.float())
        transformed_img = self.img_pad(resized_img)  
        if self.data_split=='train':

            video = self.pretrain_transform_webvid_train(transformed_img)
        else :
            video = self.img_norm(transformed_img)



        final_aug = torch.zeros([self.num_frm, 3, self.max_img_size,self.max_img_size])
        final_aug[:video.shape[0]] = video


        return final_aug,ann

    def _load_video_from_path_decord(self, video_path, height=None, width=None, start_time=None, end_time=None, fps=-1):


        try:
            if not height or not width:
                vr = VideoReader(video_path)
            else:
                vr = VideoReader(video_path, width=width, height=height)

            vlen = len(vr)

            if start_time or end_time:
                assert fps > 0, 'must provide video fps if specifying start and end time.'

                start_idx = min(int(start_time * fps), vlen)
                end_idx = min(int(end_time * fps), vlen)
            else:
                start_idx, end_idx = 0, vlen

            if self.frm_sampling_strategy == 'uniform':
                frame_indices = np.arange(start_idx, end_idx, vlen / self.num_frm, dtype=int)
            elif self.frm_sampling_strategy == 'rand':
                frame_indices = sorted(random.sample(range(vlen), self.num_frm))
            elif self.frm_sampling_strategy == 'headtail':
                frame_indices_head = sorted(random.sample(range(vlen // 2), self.num_frm // 2))
                frame_indices_tail = sorted(random.sample(range(vlen // 2, vlen), self.num_frm // 2))
                frame_indices = frame_indices_head + frame_indices_tail
            else:
                raise NotImplementedError('Invalid sampling strategy {} '.format(self.frm_sampling_strategy))

            raw_sample_frms = vr.get_batch(frame_indices)
        except Exception as e:
            print(f'{video_path} not found')
            return torch.zeros([self.num_frm, 3, self.max_img_size,self.max_img_size])

        raw_sample_frms = raw_sample_frms.permute(0, 3, 1, 2)

        return raw_sample_frms
