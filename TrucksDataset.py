from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms as transforms
from utils import get_classes_names


class Dataset(Dataset):
    def __init__(self, csv_file_path, transform):
        self.df = pd.read_csv(csv_file_path)
        classes_dic = {}
        classes_counts = {}
        classes = get_classes_names()
        for i, c in enumerate(classes):
            classes_dic[c] = i
            classes_counts[c] = len(self.df[self.df.label == c])

        self.transform = transform


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label = self.df.iloc[index, 1]
        assert label.dtype == int
        assert label == 1 or label == 0

        # video_name = self.df.iloc[index].ClipID
        # subject_id = video_name[0:6]
        # video_folder = video_name[:video_name.find('.')]
        # frames_path = join(self.root_path, subject_id, video_folder)
        frames_path = self.df.iloc[index, 0]
        data = self.transform((frames_path, self.is_train))
        return data, label

