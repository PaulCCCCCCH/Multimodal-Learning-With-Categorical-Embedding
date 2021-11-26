import os
import torch
import numpy as np
from imageio import imread
from PIL import Image
import glob
import pickle

from termcolor import colored, cprint

from preprocess import clean_text

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import torchvision.utils as vutils
from torchvision import datasets

from transformers import BertTokenizer
from preprocess import clean_text

from base_dataset import BaseDataset
from base_dataset import expand2square

from paths import dataroot

task_dict = {
    'task1': 'informative',
    'task2': 'humanitarian',
    'task2_merged': 'humanitarian',
}

labels_task1 = {
    'informative': 1,
    'not_informative': 0
}

labels_task2 = {
    'infrastructure_and_utility_damage': 0,
    'not_humanitarian': 1,
    'other_relevant_information': 2,
    'rescue_volunteering_or_donation_effort': 3,
    'vehicle_damage': 4,
    'affected_individuals': 5,
    'injured_or_dead_people': 6,
    'missing_or_found_people': 7,
}

labels_task2_merged = {
    'infrastructure_and_utility_damage': 0,
    'not_humanitarian': 1,
    'other_relevant_information': 2,
    'rescue_volunteering_or_donation_effort': 3,
    'vehicle_damage': 4,
    'affected_individuals': 5,
    'injured_or_dead_people': 5,
    'missing_or_found_people': 5,
}


class CrisisMMDataset(BaseDataset):

    def read_data(self, ann_file, category_file):
        with open(ann_file, encoding='utf-8') as f:
            self.info = f.readlines()[1:]

        if category_file is not None:
            with open(category_file, 'rb') as f:
                category_dict = pickle.load(f)

        self.data_list = []

        for l in self.info:
            l = l.rstrip('\n')
            event_name, tweet_id, image_id, tweet_text,	image,	label,	label_text,	label_image, label_text_image = l.split(
                '\t')

            path_image = '%s/%s' % (self.dataset_root, image)
            self.data_list.append(
                {
                    'path_image': path_image,

                    'text': tweet_text,
                    'text_tokens': self.tokenize(tweet_text),

                    'label_str': label,
                    'label': self.label_map[label],

                    'label_image_str': label_image,
                    'label_image': self.label_map[label_image],

                    'label_text_str': label_text,
                    'label_text': self.label_map[label_text],

                    'category_vector': 1 if category_file is None else category_dict[path_image].astype(np.float32)
                }
            )

    def tokenize(self, sentence):
        ids = self.tokenizer(clean_text(
            sentence), padding='max_length', max_length=40, truncation=True).items()
        return {k: torch.tensor(v) for k, v in ids}

    def initialize(self, opt, phase='train', cat='all', task='task2', shuffle=False, no_transform=False, use_cate=True):
        self.opt = opt
        self.shuffle = shuffle

        self.category_root = f'{dataroot}/CrisisMMD_extra'
        self.dataset_root = f'{dataroot}/CrisisMMD_v2.0_toy' if opt.debug else f'{dataroot}/CrisisMMD_v2.0'
        self.image_root = f'{self.dataset_root}/data_image'
        self.label_map = None
        self.no_transform = no_transform
        if task == 'task1':
            self.label_map = labels_task1
            task_str = task
        elif task == 'task2':
            self.label_map = labels_task2
            task_str = task
        elif task == 'task2_merged':
            self.label_map = labels_task2_merged
            task_str = 'task2'
        else:
            raise NotImplemented

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        ann_file = '%s/crisismmd_datasplit_all/task_%s_text_img_%s.tsv' % (
            self.dataset_root, task_dict[task], phase
        )

        category_file = '%s/%s_info_dict_%s.pkl' % (
            self.category_root, task_str, phase
        ) if use_cate else None

        # Append list of data to self.data_list
        self.read_data(ann_file, category_file)

        if self.shuffle:
            np.random.default_rng(seed=0).shuffle(self.data_list)
        self.data_list = self.data_list[:self.opt.max_dataset_size]
        cprint('[*] %d samples loaded.' % (len(self.data_list)), 'yellow')

        self.N = len(self.data_list)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        if self.no_transform:
            self.transforms = transforms.Compose([
                transforms.Resize((opt.load_size, opt.load_size)),
                transforms.ToTensor(),
            ])

        else:
            self.transforms = transforms.Compose([
                # transforms.Lambda(lambda img: __scale_shortside(img, opt.load_size, opt.crop_size, Image.BICUBIC)),
                # transforms.Lambda(lambda img: scale_shortside(
                #     img, opt.load_size, opt.crop_size, Image.BICUBIC)),
                transforms.Lambda(lambda img: expand2square(img)),
                transforms.Resize((opt.load_size, opt.load_size)),
                transforms.RandomHorizontalFlip(0.2),
                transforms.RandomGrayscale(0.1),
                transforms.RandomAffine(20),
                transforms.RandomCrop((opt.crop_size, opt.crop_size)),
                transforms.ToTensor(),
                transforms.ColorJitter(
                    brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

    def __getitem__(self, index):
        data = self.data_list[index]
        # if 'image' not in data:
        #     with Image.open(data['path_image']).convert('RGB') as img:
        #         image = self.transforms(img)
        #     data['image'] = image
        # return data
        to_return = {}
        for k, v in data.items():
            to_return[k] = v

        with Image.open(data['path_image']).convert('RGB') as img:
            image = self.transforms(img)
        to_return['image'] = image
        return to_return

    def __len__(self):
        return len(self.data_list)

    def name(self):
        return 'CrisisMMDataset'
