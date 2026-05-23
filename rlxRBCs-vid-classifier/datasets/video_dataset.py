''' Custom dataset for video files '''

import numpy as np
import torch
from torch.utils.data import Dataset
from pytorchvideo.data.encoded_video import EncodedVideo
from config import cfg
import pandas as pd
from pathlib import Path
import os
from labels import load_label_mapping


"""
Lightweight custom dataset with optional random temporal sampling,
integrating config.py parameters for duration and transforms

init - Initialize the Dataset object. We initialize the directory containing the images, 
the annotations file, and transform

len - Returns number of samples in our dataset

getitem -  Loads and returns a sample from the dataset at the given index idx. Based on the index, 
it identifies the image’s location on disk, retrieves the corresponding label from the csv data
"""

class RBCsDataset(Dataset):
    def __init__(
        self,
        csv_file,
        video_dir,
        transform=None,
        clip_duration=None,
        random_clip=None,
        seed=None,
        class_to_idx=None,
        filename_col: str = "filename",
        label_col: str = "label",
    ):
        """
        Args:
            csv_file (str or Path): Path to CSV file with columns ['filename', 'label']
            video_dir (str or Path): Root directory containing the video files
            transform (callable, optional): Transformations applied to the video tensor
            clip_duration (float): Maximum duration (in seconds) of the video clip to sample
            random_clip (bool): If True, sample a random start time within the video duration
            class_to_idx (dict): Mapping from class label names to integer indices
        """
        # Enforce CSV header contract: filenames and labels are required
        self.filename_col = filename_col
        self.label_col = label_col

        if class_to_idx is None:
            # Build authoritative class mapping from the training CSV (cfg.train_csv)
            # Caller should normally pass class_to_idx explicitly; if omitted we derive from csv_file.
            self.class_to_idx = load_label_mapping(csv_file, label_col=self.label_col, filename_col=self.filename_col)
        else:
            self.class_to_idx = class_to_idx

        # Read CSV robustly and normalize column names so workers see the same headers
        self.annotations = pd.read_csv(csv_file, skipinitialspace=True)
        self.annotations.columns = self.annotations.columns.str.strip()
        self.video_dir = Path(video_dir)
        self.transform = transform
        self.clip_duration = clip_duration if clip_duration is not None else cfg.clip_duration
        self.random_clip = random_clip if random_clip is not None else cfg.random_clip
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        video_path = str(self.video_dir / row[self.filename_col])
        label = self.class_to_idx[row[self.label_col]]
        
        # Load video using PyTorchVideo
        video = EncodedVideo.from_path(video_path)

        # Calculate clip duration and randomized start time
        video_duration = video.duration
        clip_duration = min(self.clip_duration, video_duration)
        start_time = 0.0

        if self.random_clip and video_duration > clip_duration:
            start_time = self.rng.uniform(0, video_duration - clip_duration)

        # Extract clip segment
        # PyTorchVideo natively returns a tensor of shape (C, T, H, W)
        video_data = video.get_clip(start_sec=start_time, end_sec=start_time + clip_duration)["video"]
        
        # Apply spatial/normalization transforms to the (C, T, H, W) tensor
        if self.transform:
            video_data = self.transform(video_data)

        return video_data, label

