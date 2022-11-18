from torch.utils.data import Dataset
import pandas as pd
from utils import get_classes_names
from torchvision.io import read_image


class TrucksDataset(Dataset):
    def __init__(self, csv_file_path, transform):
        self.df = pd.read_csv(csv_file_path)
        self.classes_dic = {}
        classes = get_classes_names()
        for i, c in enumerate(classes):
            self.classes_dic[c] = i

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label = self.df.iloc[index, 1]
        label = self.classes_dic[label]
        assert 0 <= label < 8

        path = self.df.iloc[index, 0]
        img = read_image(path)
        data = self.transform(img)
        return data, label

