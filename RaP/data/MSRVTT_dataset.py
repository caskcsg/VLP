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
from torchvision import transforms
from data.data_utils import (
    ImageResize, ImagePad, image_to_tensor)
    
decord.bridge.set_bridge("torch")
import pandas as pd 
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

        
class msrvtt7k_dataset(Dataset):

    def __init__(self, video_root,data_split, num_frm=4, frm_sampling_strategy="rand", max_img_size=384, video_fmt='.mp4'):

        if data_split=='test':
            json_file = 'meta_data/MSRVTT_7k/msrvtt_test.jsonl'
        else:
            json_file = os.path.join("meta_data/MSRVTT_7k/txt",f'{data_split}_videodatainfo.json')

        with open(json_file, "r") as f:
            self.annotation = [json.loads(l.strip("\n")) for l in f.readlines()]
        if data_split=='val':
            random.shuffle(self.annotation)
            self.annotation = self.annotation[:1000]

        self.num_frm = num_frm
        self.frm_sampling_strategy = frm_sampling_strategy
        self.max_img_size = max_img_size
        self.video_root = video_root
        self.video_fmt = video_fmt

        self.img_norm = ImageNorm(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        
        self.text = [pre_caption(ann['caption'],64) for ann in self.annotation]
        self.txt2video = [i for i in range(len(self.annotation))]
        self.video2txt = self.txt2video
            
            
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]  

        video_path = os.path.join(self.video_root, ann['clip_name'] + self.video_fmt) 

        vid_frm_array = self._load_video_from_path_decord(video_path, height=self.max_img_size, width=self.max_img_size)

        video = self.img_norm(vid_frm_array.float())

        return video,ann['caption']


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
            return None

        
        # import cv2
        raw_sample_frms = raw_sample_frms.permute(0, 3, 1, 2)


        return raw_sample_frms

class msrvtt9k_compact_dataset(Dataset):

    def __init__(self, video_root,data_split, num_frm=4, frm_sampling_strategy="rand", max_img_size=384, video_fmt='.mp4'):
        self.video_root = video_root
        json_fp = f'meta_data/MSRVTT_9k/annotation/MSR_VTT.json'
        with open(json_fp, 'r') as fid:
            data = json.load(fid)
        df = pd.DataFrame(data['annotations'])
        split_dir='meta_data/MSRVTT_9k/high-quality/structured-symlinks'

        train_list_path = "train_list_jsfusion.txt"
        test_list_path = "val_list_jsfusion.txt"
        js_test_cap_idx_path = "jsfusion_val_caption_idx.pkl"

        train_df = pd.read_csv(os.path.join(split_dir, train_list_path), names=['videoid'])
        test_df = pd.read_csv(os.path.join(split_dir, test_list_path), names=['videoid'])

        self.split=data_split
        self.split_sizes = {'train': len(train_df), 'val': len(test_df), 'test': len(test_df)}
        if self.split == 'train':
            df = df[df['image_id'].isin(train_df['videoid'])]
        else:
            df = df[df['image_id'].isin(test_df['videoid'])]

        self.metadata = df.groupby(['image_id'])['caption'].apply(list)

        if js_test_cap_idx_path is not None and self.split != 'train':
            caps = pd.Series(np.load(os.path.join(split_dir, js_test_cap_idx_path), allow_pickle=True))
            new_res = pd.DataFrame({'caps': self.metadata, 'cap_idx': caps})
            new_res['test_caps'] = new_res.apply(lambda x: [x['caps'][x['cap_idx']]], axis=1)
            self.metadata = new_res['test_caps']

        self.num_frm = num_frm
        self.frm_sampling_strategy = frm_sampling_strategy
        self.max_img_size = max_img_size
        # self.video_root = video_root
        self.video_fmt = video_fmt
        self.metadata = pd.DataFrame({'captions': self.metadata})

        self.img_norm = ImageNorm(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        self.text=[]
        self.video= []

        for index,row in self.metadata.iterrows():
            if self.split=='train':
                for caption in row['captions']:
                    self.text.append(pre_caption(caption,64))

                    self.video.append(index)
            else :
                self.text.append(pre_caption(row['captions'][0],64))
                self.video.append(index)

            
        self.txt2video = [i for i in range(len(self.text))]
        self.video2txt = self.txt2video
            
            
            
    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):

        video_path = os.path.join(self.video_root, self.video[index] + '.mp4')

        vid_frm_array = self._load_video_from_path_decord(video_path, height=self.max_img_size, width=self.max_img_size)

        video = self.img_norm(vid_frm_array.float())

        return video,self.text[index]


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
            return None
        raw_sample_frms = raw_sample_frms.permute(0, 3, 1, 2)
        return raw_sample_frms
