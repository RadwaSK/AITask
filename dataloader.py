import torch, torchvision
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

print('PyTorch version:', torch.__version__)
print('TorchVision version:', torchvision.__version__)


def get_train_val_dataloader(train_path='dataset/train', val_ratio=0.2, batch_size=4, output_size=224):
    train_transformer = transforms.Compose([
        transforms.Resize((output_size, output_size)),
        transforms.RandomHorizontalFlip(0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transformer = transforms.Compose([
        transforms.Resize((output_size, output_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # I made two different datasets from the same folder to apply different transformers to them
    train_dataset = datasets.ImageFolder(train_path, transform=train_transformer)
    valid_dataset = datasets.ImageFolder(train_path, transform=val_transformer)

    all_data_size = len(train_dataset)
    val_size = int(val_ratio * all_data_size)
    indices = list(range(all_data_size))

    train_idx, val_idx = indices[val_size:], indices[:val_size]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, drop_last=False)
    val_dataloader = DataLoader(valid_dataset, batch_size=batch_size, sampler=val_sampler, drop_last=False)

    return train_dataloader, val_dataloader


def get_test_dataloader(test_path='dataset/train', val_ratio=0.2, batch_size=4, output_size=224):
    test_transformer = transforms.Compose([
        transforms.Resize((output_size, output_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(test_path, transform=test_transformer)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    return test_dataloader

