import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from vocab_preprocessing import Vocabulary
from pycocotools.coco import COCO

def collate_fn2(data):
    images, captions = zip(*data)
    images = torch.stack(images, 0)
    return images, captions

class CocoTrainDataset(data.Dataset):
    def __init__(self, root, json, vocab, transform=None):
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    images = torch.stack(images, 0)
    lengths = torch.tensor([len(cap) for cap in captions]).long()
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

def get_train_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    coco = CocoTrainDataset(root=root,
                       json=json,
                       vocab=vocab,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return train_loader



class CocoValidationDataset(data.Dataset):
    def __init__(self, root, json, vocab, transform=None):
        self.root = root
        self.vocab = vocab
        self.transform = transform

        self.coco = COCO(json)
        ids = list(self.coco.anns.keys())
        captions = {}
        for i in ids:
            im_id = self.coco.anns[i]['image_id']
            if im_id not in captions:
                captions[im_id] = []
            captions[im_id].append(i)
            
        self.ids = list(captions.keys())
        self.captions = captions


    def __getitem__(self, idx):
        coco = self.coco
        vocab = self.vocab
        
        img_id = self.ids[idx]
        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        captions = []
        for ann_id in self.captions[img_id]:
            caption = coco.anns[ann_id]['caption']
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = []
            caption.append(vocab('<start>'))
            caption.extend([vocab(token) for token in tokens])
            caption.append(vocab('<end>'))
            captions.append(caption)
        return image, captions

    def __len__(self):
        return len(self.ids)

def get_validation_loader(root, json, vocab, transform, batch_size, num_workers):
    coco = CocoValidationDataset(root=root,
                                 json=json,
                                 vocab=vocab,
                                 transform=transform)
    validation_loader = torch.utils.data.DataLoader(dataset=coco, 
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    collate_fn=collate_fn2)  # Use the globally defined collate_fn2
    return validation_loader