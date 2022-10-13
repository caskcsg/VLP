import json
import os
import random

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import torch

from abc import abstractmethod
from torchvision import transforms as tv_transforms
from data.utils import pre_caption
import decord
import pandas as pd
import numpy as np


def get_video_len(video_path):
    cap = cv2.VideoCapture(video_path)
    if not (cap.isOpened()):
        return False
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return vlen

def sample_frames(num_frames, vlen, sample='rand', fix_start=None):

    if sample=='uniform':
        frame_indices = np.arange(0, vlen, vlen / num_frames, dtype=int)
    elif sample == 'headtail':
        frame_indices_head = sorted(random.sample(range(vlen // 2), num_frames // 2))
        frame_indices_tail = sorted(random.sample(range(vlen // 2, vlen), num_frames // 2))
        frame_indices = frame_indices_head + frame_indices_tail
    else:
        raise NotImplementedError

    return frame_indices

def read_frames_decord(video_path, num_frames, sample='rand', fix_start=None):

    video_reader = decord.VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = video_reader.get_batch(frame_idxs)
    frames = torch.from_numpy(frames.asnumpy()).float() / 255
    frames = frames.permute(0, 3, 1, 2)

    return frames,frame_idxs
def read_frames_decord_eval(video_path, num_frames, sample='rand', fix_start=None):

    video_reader = decord.VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = video_reader.get_batch(frame_idxs)
    # frames = torch.from_numpy(frames.asnumpy()).float() / 255
    frames = frames.float() / 255
    frames = frames.permute(0, 3, 1, 2)

    return frames,frame_idxs

video_reader = {
    'decord': read_frames_decord,
    'eval_decord':read_frames_decord_eval
}

class pretrain_webvid_dataset(Dataset):
    def __init__(self, config, transform, max_words=30,data_split='train'):        

        self.dataset_name = config['args']['dataset_name']
        self.text_params = config['args']['text_params']
        self.video_params = config['args']['video_params']
        self.data_dir = config['args']['data_dir']
        self.metadata_dir = self.data_dir
        self.split = data_split
        self.cut = config['args']['cut']
        self.subsample = config['args']['subsample']
        self.video_reader = video_reader['decord']
        self.label_type = 'caption'
        self.frame_sample= config['args']['video_params']['frame_sample']
        self.sliding_window_stride=-1
        self._load_metadata() # 
        if config['use_subset'] == 1 :
            self.metadata=self.metadata.sample(frac = 0.25)

        self.transforms = transform
        
    @abstractmethod
    def _get_video_path(self, sample):
        rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        full_video_fp = os.path.join(self.data_dir, 'videos', rel_video_fp)
        return full_video_fp, rel_video_fp

    def _get_caption(self, sample):
        return sample['caption']
    @abstractmethod
    def _load_metadata(self):
        metadata_dir = os.path.join(self.metadata_dir, 'metadata')
        metadata_fp = os.path.join(metadata_dir, f'results_{self.cut}_{self.split}.csv')
        metadata = pd.read_csv(metadata_fp)

        if self.subsample < 1:
            metadata = metadata.sample(frac=self.subsample)
        elif self.split == 'val':
            metadata = metadata.sample(1000, random_state=0)  # 15k val is unnecessarily large, downsample.

        metadata['caption'] = metadata['name']
        del metadata['name']
        self.metadata = metadata
        # TODO: clean final csv so this isn't necessary
        self.metadata.dropna(inplace=True)
        self.metadata['caption'] = self.metadata['caption'].str[:350]
    def _get_video_lens(self):
        vlen_li = []
        for idx, row in self.metadata.iterrows():
            video_path = self._get_video_path(row)[0]
            vlen_li.append(get_video_len(video_path))

        return vlen_li

    def _fix_temporal_samples(self):
        self.metadata['vlen'] = self._get_video_lens()
        self.metadata['frame_intervals'] = self.metadata['vlen'].apply(
            lambda x: np.linspace(start=0, stop=x, num=min(x, self.video_params['num_frames']) + 1).astype(int))
        self.metadata['fix_start'] = self.metadata['frame_intervals'].apply(
            lambda x: np.arange(0, int(x[-1] / len(x - 1)), self.sliding_window_stride)
        )
        self.metadata = self.metadata.explode('fix_start')

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, item):
        
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, rel_fp = self._get_video_path(sample)
        caption = self._get_caption(sample)

        video_loading = self.video_params.get('loading', 'strict')
        frame_sample = self.frame_sample

        fix_start = None
        if self.split == 'test' or self.split == 'val':
            frame_sample = 'uniform'

        if self.sliding_window_stride != -1:
            fix_start = sample['fix_start']
        try:
            if os.path.isfile(video_fp):
                imgs, idxs = self.video_reader(video_fp, self.video_params['num_frames'], frame_sample,
                                               fix_start=fix_start)
            else:
                print(f"Warning: missing video file {video_fp}.")
                assert False
        except Exception as e:
            if video_loading == 'strict':
                raise ValueError(
                    f'Video loading failed for {video_fp}, video loading for this dataset is strict.') from e
            else:
                imgs = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))
                imgs = tv_transforms.ToTensor()(imgs).unsqueeze(0)

        if self.transforms is not None:
            
            imgs_ori = self.transforms(imgs)
            imgs_aug = self.transforms(imgs)
        final = torch.zeros([self.video_params['num_frames'], 3, self.video_params['input_res'],
                             self.video_params['input_res']])
        final[:imgs_ori.shape[0]] = imgs_ori
        
        final_aug = torch.zeros([self.video_params['num_frames'], 3, self.video_params['input_res'],
                             self.video_params['input_res']])
        final_aug[:imgs_aug.shape[0]] = imgs_aug
        

        return final,final_aug,caption


class pretrain_cc3m_video_dataset(Dataset):
    def __init__(self, ann_file, laion_path, transform,num_frames,config): 
        self.num_frames=num_frames
        self.annotation = []
        for f in ann_file:
            print('loading '+f)
            ann = json.load(open(f,'r'))
            self.annotation += ann
        if config['use_subset'] == 1:
            random.shuffle(self.annotation)
            self.annotation =self.annotation[: int(len(self.annotation)/4)]
        self.transform = transform
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    

        ann = self.annotation[index]

        image = Image.open(ann['image']).convert('RGB')
        image_ori = self.transform(image)
        image_aug = self.transform(image)
        caption = pre_caption(ann['caption'],30)
        image_ori = image_ori.unsqueeze(0).repeat(self.num_frames,1,1,1)
        image_aug = image_aug.unsqueeze(0).repeat(self.num_frames,1,1,1)

        return image_ori,image_aug, caption