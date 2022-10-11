from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pickle
from data.rawvideo_util import RawVideoExtractor
from data.utils import pre_caption
import torch
import random
class ImageNorm(object):
    """Apply Normalization to Image Pixels on GPU
    """
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        
    def __call__(self, img):

        if torch.max(img) > 1 and self.mean.max() <= 1:
            img.div_(255.)
        return img.sub_(self.mean).div_(self.std)


class msvd_dataset(Dataset):
    """MSVD dataset loader."""
    def __init__(
            self,
            video_root,
            data_split,
            num_frm=8,
            max_img_size=256,
            feature_framerate=1,
            frame_order=0,
            slice_framepos=2,
    ):
        self.data_split =data_split
        subset=data_split
        features_path=video_root
        data_path='meta_data/MSVD'
        self.data_path = data_path
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_frames = num_frm
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.subset = subset
        assert self.subset in ["train", "val", "test"]
        video_id_path_dict = {}
        video_id_path_dict["train"] = os.path.join(self.data_path, "train_list.txt")
        video_id_path_dict["val"] = os.path.join(self.data_path, "val_list.txt")
        video_id_path_dict["test"] = os.path.join(self.data_path, "test_list.txt")
        caption_file = os.path.join(self.data_path, "raw-captions.pkl")

        with open(video_id_path_dict[self.subset], 'r') as fp:
            video_ids = [itm.strip() for itm in fp.readlines()]

        with open(caption_file, 'rb') as f:
            captions = pickle.load(f)

        video_dict = {}
        for root, dub_dir, video_files in os.walk(self.features_path):
            for video_file in video_files:
                video_id_ = ".".join(video_file.split(".")[:-1])
                if video_id_ not in video_ids:
                    continue
                file_path_ = os.path.join(root, video_file)
                video_dict[video_id_] = file_path_
        self.video_dict = video_dict
        
        self.txt2video = [ ]
        
        self.text = []
        self.sample_len = 0
        self.sentences_dict = {}
        self.cut_off_points = []
        
        video_index =0 
        for video_id in video_ids:
            assert video_id in captions
            for cap in captions[video_id]:
                cap_txt = " ".join(cap)
                self.text.append(cap_txt)
                self.txt2video.append(video_index)
                # break 
            video_index+=1

        for video_id in video_ids:
            assert video_id in captions
            temp_txt = []
            for cap in captions[video_id]:
                cap_txt = " ".join(cap)
                temp_txt.append(cap_txt)
            self.sentences_dict[len(self.sentences_dict)] = (video_id, temp_txt)
            # self.cut_off_points.append(len(self.sentences_dict))
        self.video2txt = [i for i in range(len(video_ids))]

        # import ipdb;ipdb.set_trace()

        self.img_norm = ImageNorm(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        self.sample_len = len(self.sentences_dict)
        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=max_img_size)


    def __len__(self):
        return self.sample_len
        # return 128

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)

        for i, video_id in enumerate(choice_video_ids):
            video_path = self.video_dict[video_id]

            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            
            raw_video_data = raw_video_data['video']
            

            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data

                # L x T x 3 x H x W
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, idx):
        
        video_id, ls_caption = self.sentences_dict[idx]
        caption = random.choice(ls_caption)
        
        self.video_dict[video_id]
        video, video_mask = self._get_rawvideo([video_id])
        
        video = video.squeeze()

        video = torch.from_numpy(video)

        return video.float(), caption
