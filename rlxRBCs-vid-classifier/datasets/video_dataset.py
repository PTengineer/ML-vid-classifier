''' Custom dataset for video files '''

import torch
from torch.utils.data import Dataset
from pytorchvideo.data.encoded_video import EncodedVideo
import pandas as pd
import os


"""
Custom dataset class of short clips and annotations 

init - Initialize the Dataset object. We initialize the directory containing the images, 
the annotations file, and transform

len - Returns number of samples in our dataset

getitem -  Loads and returns a sample from the dataset at the given index idx. Based on the index, 
it identifies the image’s location on disk, retrieves the corresponding label from the csv data
"""

class RBCsDataset(Dataset):
    def __init__(self, csv_file, video_dir, transform=None, clip_duration=5.0):
        self.annotations = pd.read_csv(csv_file)
        self.video_dir = video_dir
        self.transform = transform
        self.clip_duration = clip_duration
        self.classes = sorted(self.annotations['label'].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        video_path = os.path.join(self.video_dir, row['filename'])
        label = self.class_to_idx[row['label']]
        video = EncodedVideo.from_path(video_path)
        video_data = video.get_clip(start_sec=0, end_sec=self.clip_duration)["video"]
        if self.transform:
            video_data = self.transform(video_data)
        return video_data, label
