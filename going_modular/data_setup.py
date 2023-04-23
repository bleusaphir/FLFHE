"""
Contains functionality for creating PyTorch DataLoaders for image classification data.
"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

from .common import *


NORMALIZE_DICT = {
        'mnist': dict(mean=(0.1307,), std=(0.3081,)),
        'cifar': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        'animaux': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    }


# Define model, architecture and dataset
# The DataLoaders downloads the training and test data that are then normalized.
def load_datasets(num_clients: int, batch_size: int, resize: int, seed: int, num_workers: int, splitter=10,
                  dataset="cifar", data_path="./data/"):
    # Download and transform CIFAR-10 (train and test)

    list_transforms = [transforms.ToTensor(), transforms.Normalize(**NORMALIZE_DICT[dataset])]

    if dataset == "cifar":
        transformer = transforms.Compose(
            list_transforms
        )

        trainset = CIFAR10(data_path + dataset, train=True, download=True, transform=transformer)
        testset = CIFAR10(data_path + dataset, train=False, download=True, transform=transformer)

    else:
        transformer = transforms.Compose([transforms.Resize((resize, resize))] + list_transforms)
        supp_ds_store(data_path + dataset)
        supp_ds_store(data_path + dataset+"/train")
        supp_ds_store(data_path + dataset + "/test")
        trainset = datasets.ImageFolder(data_path + dataset+"/train", transform=transformer)
        testset = datasets.ImageFolder(data_path + dataset+"/test", transform=transformer)

    print(f"The training set is created for the classes : {trainset.classes}")
    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(trainset) // num_clients
    lengths = [partition_size] * (num_clients-1)
    lengths += [len(trainset) - sum(lengths)]
    datasets_train = random_split(trainset, lengths, torch.Generator().manual_seed(seed))
    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets_train:
        len_val = int(len(ds) * splitter/100)  # splitter % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        print("data split : ", lengths)
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(seed))
        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size))

    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloaders, valloaders, testloader
