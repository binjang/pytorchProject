import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, file):
        #TODO: 파일 종류별 데이터 처리
        if csv:
            self.label = pd.read_csv(file)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        sample = torch.tensor

if __name__ == "__main__":
    dataset = CustomDataset()
    dataset = DataLoader(dataset, batch_size=4, shuffle = True)