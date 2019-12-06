import torch
import torchvision as tv
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.utils import save_image
from tqdm import tqdm

import os
import matplotlib.pyplot as plt


class Autoencoder(nn.Module):
    """
    Autoencoder model implemented with Linear layers.
    """
    def __init__(self):
        super(Autoencoder,self).__init__()

        self.autouenc = nn.Sequential()
        self.encoder  = nn.Linear(28*28, 64)
        self.decoder  = nn.Linear(64, 28*28)

        self.autouenc.add_module('enc',self.encoder)
        self.autouenc.add_module('dec',self.decoder)

    def forward(self, x):
        pred = F.relu(self.encoder(x))
        pred = F.relu(self.decoder(pred))
        return pred


def get_train_valid_loader(data_dir,
                           batch_size,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg
 
    transform = transforms.Compose([transforms.ToTensor()])

    # load the dataset
    train_dataset = datasets.MNIST(
        root=data_dir, train=True,
        download=True, transform=transform,
    )

    valid_dataset = datasets.MNIST(
        root=data_dir, train=True,
        download=True, transform=transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return (train_loader, valid_loader)

if __name__ == "__main__":
    # Get train/dev partition
    train_data, dev_data =get_train_valid_loader('./MNIST', 64, 0)

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        model = Autoencoder().cuda()
    else:
        model = Autoencoder()

    # Mean Squared Error Loss
    distance = nn.MSELoss()
    # alpha = 0.001 by default
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-5)

    # TODO make the amount of epochs a parameter
    for epoch in range(20):
        # TODO make this cuda or cpu depending on torch.cuda.is_available()
        train_loss = torch.cuda.FloatTensor()
        dev_loss   = torch.cuda.FloatTensor()

        for data in train_data:
            img, _ = data
            # TODO make this cuda or cpu depending on torch.cuda.is_available()
            img = Variable(img.view(-1, 28*28)).cuda()
            # ===================forward=====================
            output = model(img)
            assert(output.shape == img.shape)
            loss = distance(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ===================log========================
            train_loss = torch.cat([train_loss, loss.data.reshape([1])])

        # ==================dev=======================
        with torch.no_grad():
            for data in dev_data:
                img, _ = data
                # TODO make this cuda or cpu depending on torch.cuda.is_available()
                img = Variable(img.view(-1, 28*28)).cuda()
                # ===================forward=====================
                output = model(img)
                loss = distance(output, img)
                dev_loss = torch.cat([dev_loss, loss.data.reshape([1])])

        print('epoch [{}/{}], train loss:{:.4f}, dev loss:{:.4f}'.format(epoch+1, 20, torch.mean(train_loss), torch.mean(dev_loss)))

        torch.save(model.state_dict(),
                   "Autoenc_models/Autoencoder{}".format(epoch))
