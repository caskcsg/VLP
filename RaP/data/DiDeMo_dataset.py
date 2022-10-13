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

class didemo_dataset(Dataset):
    def __init__(self, video_root,data_split, num_frm=4, frm_sampling_strategy="rand", max_img_size=384, video_fmt='.avi'):

        self.text = []
        self.vedio_id = []
        lines = open(f'meta_data/DiDeMo/DiDeMo_{data_split}.tsv').readlines()
        for line in lines:
            ls_now = line.strip().split('\t')
            if len(ls_now)!=2:
                continue
            try:
                if '12090392@N02_13482799053_87ef417396' in ls_now[1]:
                    continue
                self.text.append(pre_caption(ls_now[0],64))
                self.vedio_id.append(ls_now[1])
            except:
                import ipdb;ipdb.set_trace()

        self.data_split = data_split
        self.num_frm = num_frm
        self.frm_sampling_strategy = frm_sampling_strategy
        self.max_img_size = max_img_size
        self.video_root = video_root
        self.video_fmt = video_fmt

        self.img_resize = ImageResize(
            max_img_size,
            "bilinear")
        self.img_pad = ImagePad(
            max_img_size, max_img_size) 


        self.img_norm = ImageNorm(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

        self.txt2video = [i for i in range(len(self.text))]
        self.video2txt = self.txt2video
            

    def __len__(self):
        return len(self.text)


    def __getitem__(self, index):

        ann = self.text[index]
        if self.data_split=='train':

            ls_now = ann.split(',')
            ann = random.choice(ls_now)
            ann=ann.strip()

        vedio_id = self.vedio_id[index]

        video_path = os.path.join(f'{self.video_root}',vedio_id) 

        try :
            vid_frm_array = self._load_video_from_path_decord(video_path, height=self.max_img_size, width=self.max_img_size)
        except:

            print(f'{video_path} not not not not found')
            item_new = random.randint(1, self.__len__())
            return self.__getitem__(item_new)


        resized_img = self.img_resize(vid_frm_array.float())
        transformed_img = self.img_pad(resized_img) 

        video = self.img_norm(transformed_img)

        video = transformed_img

        final_aug = torch.zeros([self.num_frm, 3, self.max_img_size,self.max_img_size])
        final_aug[:video.shape[0]] = video


        return final_aug,ann

    def _load_video_from_path_decord(self, video_path, height=None, width=None, start_time=None, end_time=None, fps=-1):

        all_format =['mov', 'mp4', 'avi', 'wmv', 'm4v', 'mpg', '3g2', '3gp', 'mts', '', 'mpeg', '3gpp', 'm2ts', 'py', 'qt', 'asf', 'divx', 'out', 'null']

        all_index=0
        while all_index<len(all_format):
            new_path = video_path.split('.')[0]+'.'+all_format[all_index]
            if(os.path.exists(new_path)):
                break
            all_index+=1
        assert all_index<len(all_format),f'{video_path} read fail'

        video_path = new_path

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

        raw_sample_frms = raw_sample_frms.permute(0, 3, 1, 2)

        return raw_sample_frms