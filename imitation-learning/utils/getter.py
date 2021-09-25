from datasets import *
from models import *
from losses import *
from utils import *

from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path

import argparse
import numpy as np
import json
import os
import random
import torch
import torch.nn.functional as F
import torch.optim as optim

def get_dataloader_from_json(args):
    obses, samples = create_dataset_from_json(args.episode_dir)
    print('obses:', len(obses), '- samples:', len(samples))

    labels = [sample[-1] for sample in samples]
    actions = ['north', 'south', 'west', 'east', 'bcity']
    for value, count in zip(*np.unique(labels, return_counts=True)):
        print(f'{actions[value]:^5}: {count:>3}')

    train_samples, val_samples= train_test_split(samples, test_size=args.split_ratio, random_state=args.seed_value, stratify=labels)

    train_loader = DataLoader(
        LuxDataset(obses, train_samples), 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2
    )
    val_loader = DataLoader(
        LuxDataset(obses, val_samples), 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2
    )

    return {"train": train_loader, "val": val_loader}