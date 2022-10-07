import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision import  datasets

def get_tiny_imagenet(data_dir='tiny-imagenet-200',subset='train'):
    DATA_DIR=data_dir
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VALID_DIR = os.path.join(DATA_DIR, 'val')
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    if subset=='train':
        data=TRAIN_DIR
    elif subset=='val':
        data=VALID_DIR
    elif subset=='test':
        data=TEST_DIR
    else:
        raise ValueError(f"subset {subset} is not defined!")
    
    transform_default=T.Compose([
        T.Resize(32),
        T.RandomHorizontalFlip(),
        T.ToTensor()
    ])
    transform_test=T.Compose([
        T.Resize(32),
        T.ToTensor()
    ])

    dataset = datasets.ImageFolder(data, transform=(transform_default if subset=='train' else transform_test))
    
    return dataset

def get_dataloader(data_name:str,
                   data_dir:str,
                   train:bool=True,
                   subset='train',
                   batch_size:int=1000,
                   use_cuda:bool=torch.cuda.is_available(),
                   use_default_transform:bool=True
                   )->tuple[DataLoader,int]:
    transform_cifar=T.Compose([
        T.RandomResizedCrop(32),
        T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
        T.ToTensor()
    ])
    transform_mnist=T.Compose([
        T.Resize(32),
        T.RandomChoice([
                T.RandomAffine((-60, 60)),
                T.RandomAffine(0, translate=(0.2, 0.4)),
                T.RandomAffine(0, scale=(0.8, 1.1)),
                T.RandomAffine(0, shear=(-20, 20))]),
        T.ToTensor()
    ])
    transform_default=T.Compose([
        T.RandomCrop(32, padding=8),
        T.RandomHorizontalFlip(),
        T.ToTensor()
    ])
    transform_test=T.Compose([
        T.Resize(32),
        T.ToTensor()
    ])
    if data_name=='mnist':
        dataset=datasets.MNIST(root=data_dir,train=train,
                               transform=((transform_default if use_default_transform else transform_mnist) if train else transform_test),
                               download=True)
        num_classes=10
    elif data_name=='cifar10':
        data_dir=os.path.join(data_dir,'cifar10')
        dataset=datasets.CIFAR10(root=data_dir,train=train,
                                 transform=((transform_default if use_default_transform else transform_cifar) if train else transform_test),
                                 download=True)
        num_classes=10
    elif data_name=='cifar100':
        data_dir=os.path.join(data_dir,'cifar100')
        dataset=datasets.CIFAR100(root=data_dir,train=train,
                                  transform=((transform_default if use_default_transform else transform_cifar) if train else transform_test),
                                  download=True)
        num_classes=100
    elif data_name=='tiny_imagenet':
        train=(subset=='train')
        dataset=get_tiny_imagenet(data_dir,subset)
        num_classes=200
    else:
        raise ValueError(f"Dataset {data_name} is not defined!")
    
    if use_cuda:
        kwargs = {"pin_memory": True, "num_workers": 2*torch.cuda.device_count()}
    else:
        kwargs = {}
    dataloader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=train,**kwargs)
    return (dataloader,num_classes)
