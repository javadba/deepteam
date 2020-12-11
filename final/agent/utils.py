import numpy as np

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from . import dense_transforms

LABEL_NAMES = [0., 1.]

RESCUE_TIMEOUT = 30
TRACK_OFFSET = 15
DATASET_PATH = '2x2x1250z'


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor()):
        from PIL import Image
        from glob import glob
        from os import path
        self.data = []
        for f in glob(path.join(dataset_path, '*.csv')):
            i = Image.open(f.replace('.csv', '.png'))
            i.load()
            label = np.loadtxt(f,delimiter=',')[-1:]
            label_id = LABEL_NAMES.index(label)
            self.data.append((i, label_id))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = self.transform(*data)
        return data


def load_data(dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor(), num_workers=0, batch_size=64):
    dataset = SuperTuxDataset(dataset_path, transform=transform)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

