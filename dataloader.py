from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from TrucksDataset import TrucksDataset

def get_train_val_dataloader(train_csv_path='dataset/train.csv', val_csv_path='dataset/val.csv', batch_size=4, input_size=224):
    train_transformer = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transformer = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = TrucksDataset(train_csv_path, train_transformer)
    valid_dataset = TrucksDataset(val_csv_path, val_transformer)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, drop_last=False, shuffle=True)
    val_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=4, drop_last=False, shuffle=True)

    return {'train': train_dataloader, 'validation': val_dataloader}


def get_test_dataloader(test_csv_path='dataset/test_csv',  batch_size=4, input_size=224):
    test_transformer = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = TrucksDataset(test_csv_path, test_transformer)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=False)

    return test_dataloader

