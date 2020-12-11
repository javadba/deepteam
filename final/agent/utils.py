import numpy as np

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from . import dense_transforms

RESCUE_TIMEOUT = 30
TRACK_OFFSET = 15
DATASET_PATH = '/content/drive/MyDrive/2x2x1250z'
CSV_PATH = '/content/drive/MyDrive/2x2x1250z'


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor(),
                 csv_path=CSV_PATH, ):
        from PIL import Image
        from glob import glob
        from os import path
        self.data = []
        k=0
        cntr = 0
        LoadCntr = 1000
        for i,f in enumerate(glob(path.join(csv_path, '*.csv'))):
            fn = f"{dataset_path}/{f[f.rfind('/')+1:].replace('.csv', '.png')}"
            if i % LoadCntr == 0:
                print(f'Loading file[{i}] {fn} ..')
            i = Image.open(fn)
            i.load()
            self.data.append((i, np.loadtxt(f, dtype=np.float32, delimiter=',')[-1:]))
            k += 1
            if k>20000:
              break
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = self.transform(*data)
        return data


def load_data(dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor(), num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path, transform=transform)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

def load_data2(dataset_path=DATASET_PATH, csv_path=CSV_PATH, transform=dense_transforms.ToTensor(), num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path, transform=transform, csv_path=csv_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)
