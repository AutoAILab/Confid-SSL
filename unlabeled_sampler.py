from typing import Callable

import pandas as pd
import torch
import torch.utils.data
import torchvision
import json
import numpy as np
import math

class Unlabeled_ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices: list = None, num_samples: int = None, callback_get_label: Callable = None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        z = open("dict_avgconf.txt", "r")
        k = z.read()
        dict_avgconf = json.loads(k)
        z.close()

        z = open("dict_highconf_indx.txt", "r")
        k = z.read()
        dict_highconf_indx = json.loads(k)
        z.close()
        
        z = open("dict_logits.txt", "r")
        k = z.read()
        dict_logits = json.loads(k)
        z.close()
        
        z = open("current_epoch.txt", "r")
        k = z.read()
        current_epoch = json.loads(k)
        z.close()

        weights = np.ones(df["label"].shape[0])
        dict_logits = np.array(dict_logits)
        
        
        
#         soft_ratio = math.exp(-3 * (1 - min(current_epoch/500.0, 1))**3)*0.5 + 0.5
        soft_ratio = 2-2/(1+current_epoch/500)

        for current_label in range(label_to_count.shape[0]):
            if '%d' %current_label in dict_highconf_indx:
                weights[dict_highconf_indx['%d' %current_label]] = (1 - soft_ratio*dict_avgconf['%d' %current_label])
            if '%d_low' %current_label in dict_highconf_indx:
                weights[dict_highconf_indx['%d_low' %current_label]] = (2 - soft_ratio*dict_logits[dict_highconf_indx['%d_low' %current_label]])


        self.weights = torch.DoubleTensor(weights.tolist())

        # print(1)

            # weights = 1.0 / label_to_count[df["label"]]
        #
        # self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.label.squeeze(-1)
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
