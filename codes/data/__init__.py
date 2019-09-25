'''create dataset and dataloader'''
import logging
import torch.utils.data
from data.LR_dataset import LRDataset as D

def create_dataloader(dataset, dataset_opt):
    '''create dataloader '''
    return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, 
           num_workers=1, pin_memory=True)

def create_dataset(dataset_opt):
    '''create dataset'''
    dataset = D(dataset_opt)
    return dataset
