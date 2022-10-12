import json

from torch.utils.data import Dataset


class MarriagesWikidataDataset(Dataset):
    def __init__(self, filepath):
        self.data = json.load(open(filepath, 'r'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def summary(self):
        return "multi_label_dataset", "multi_label_dataset"
