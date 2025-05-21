# Create a new file: datasets/volleyball.py
from pathlib import Path
import json
import os
import torch
import numpy as np
import random
from PIL import Image
import copy
import datasets.transforms as T
from models.structures import Instances

class VolleyballDataset:
    def __init__(self, args, data_txt_path, seqs_folder, transform):
        self.args = args
        self.transform = transform
        self.num_frames_per_batch = max(args.sampler_lengths)
        self.sample_mode = args.sample_mode
        self.sample_interval = args.sample_interval
        self.video_dict = {}
        self.mot_path = args.mot_path
        
        # Load annotation structure (similar to DanceTrack)
        self.labels_full = defaultdict(lambda: defaultdict(list))
        
        # Add volleyball dataset folder
        volleyball_dir = "Volleyball"
        print(f"Adding {volleyball_dir}")
        for vid in os.listdir(os.path.join(self.mot_path, volleyball_dir)):
            vid_path = os.path.join(volleyball_dir, vid)
            
            # Load your annotations (modify according to your format)
            gt_path = os.path.join(self.mot_path, vid_path, 'gt', 'gt.txt')
            for l in open(gt_path):
                t, i, *xywh, label = l.strip().split(',')
                t, i, label = map(int, (t, i, label))
                
                # Map different object types (players=0, ball=1, court=2)
                x, y, w, h = map(float, xywh)
                self.labels_full[vid_path][t].append([x, y, w, h, i, False, label])
                
        # Set up indices for training
        vid_files = list(self.labels_full.keys())
        self.indices = []
        self.vid_tmax = {}
        
        for vid in vid_files:
            self.video_dict[vid] = len(self.video_dict)
            t_min = min(self.labels_full[vid].keys())
            t_max = max(self.labels_full[vid].keys()) + 1
            self.vid_tmax[vid] = t_max - 1
            for t in range(t_min, t_max - self.num_frames_per_batch):
                self.indices.append((vid, t))
        
        print(f"Found {len(vid_files)} videos, {len(self.indices)} frames")
        
        # Load detection database
        if args.det_db:
            with open(os.path.join(args.mot_path, args.det_db)) as f:
                self.det_db = json.load(f)
        else:
            self.det_db = defaultdict(list)
            
    # Implement other methods similar to dance.py
    # _pre_single_frame, __getitem__, etc.
    
    def _pre_single_frame(self, vid, idx: int):
        img_path = os.path.join(self.mot_path, vid, 'img1', f'{idx:08d}.jpg')
        img = Image.open(img_path)
        targets = {}
        w, h = img._size
        
        # Create frame targets with specific volleyball classes
        targets['dataset'] = 'Volleyball'
        targets['boxes'] = []
        targets['iscrowd'] = []
        targets['labels'] = []  # Use this for object types (player=0, ball=1, court=2)
        targets['obj_ids'] = []
        targets['scores'] = []
        targets['image_id'] = torch.as_tensor(idx)
        targets['size'] = torch.as_tensor([h, w])
        targets['orig_size'] = torch.as_tensor([h, w])
        
        # Add ground truth objects
        for *xywh, id, crowd, label in self.labels_full[vid][idx]:
            targets['boxes'].append(xywh)
            targets['iscrowd'].append(crowd)
            targets['labels'].append(label)  # Use the label from annotations
            targets['obj_ids'].append(id)
            targets['scores'].append(1.)
            
        # Add D-FINE detections
        txt_key = os.path.join(vid, 'img1', f'{idx:08d}.txt')
        for line in self.det_db[txt_key]:
            *box, s, label = map(float, line.split(','))
            targets['boxes'].append(box)
            targets['scores'].append(s)
            targets['labels'].append(int(label))
            
        # Convert to tensors
        targets['iscrowd'] = torch.as_tensor(targets['iscrowd'])
        targets['labels'] = torch.as_tensor(targets['labels'])
        targets['obj_ids'] = torch.as_tensor(targets['obj_ids'], dtype=torch.float64)
        targets['scores'] = torch.as_tensor(targets['scores'])
        targets['boxes'] = torch.as_tensor(targets['boxes'], dtype=torch.float32).reshape(-1, 4)
        targets['boxes'][:, 2:] += targets['boxes'][:, :2]  # Convert xywh to xyxy
        
        return img, targets

# Update datasets/__init__.py to include volleyball dataset
def build(image_set, args):
    root = Path(args.mot_path)
    assert root.exists(), f'provided path {root} does not exist'
    transform = build_transform(args, image_set)
    
    if image_set == 'train':
        data_txt_path = args.data_txt_path_train
        dataset = VolleyballDataset(args, data_txt_path, root, transform)
    if image_set == 'val':
        data_txt_path = args.data_txt_path_val
        dataset = VolleyballDataset(args, data_txt_path, root, transform)
        
    return dataset