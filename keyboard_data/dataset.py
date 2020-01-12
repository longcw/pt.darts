import os
import csv
import numpy as np
import logging
import pprint
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

LABELS = [
    "foreign_material",
    "hook_broken",
    "muk_overlay",
    "overspray",
    "pass",
    "pf_abnormal",
    "pin_broken",
    "snap_broken",
    "snap_not_engage",
]

# IMAGE_SIZE = (80, 80)
IMAGE_SIZE = (32, 32)


class KeyboardImageDataset(Dataset):
    def __init__(
        self,
        data_dir,
        subset="train",
        transform=None,
        data_balance_rate=0.0,
        keep_size=True,
        seed=1234,
    ):
        self.data_dir = data_dir
        self.subset = subset
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        if transform is not None:
            self.transform = transforms.Compose([transform, self.transform])

        # load data
        self.image_root = os.path.join(
            self.data_dir, self.subset, "{}_img".format(self.subset)
        )
        self.anno_file = os.path.join(
            self.data_dir, self.subset, "{}_annotation.csv".format(self.subset)
        )
        with open(self.anno_file, "r") as f:
            reader = csv.reader(f, delimiter=",")
            self.annoatation_header = next(reader)
            self.annotations = list(reader)
        self.cat2id = {label: i for i, label in enumerate(LABELS)}

        # init index
        all_index = list(range(len(self.annotations)))

        # data balance
        np.random.seed(seed)
        # all category contains the same number of images if balance_rate = 1
        if self.subset == "train" and data_balance_rate > 0:
            label_to_indices = {}
            for i, label in zip(all_index, self.annotations):
                label_to_indices.setdefault(label[1], [])
                label_to_indices[label[1]].append(i)

            max_length = max([len(indices) for indices in label_to_indices.values()])
            for label, indices in label_to_indices.items():
                n_sample = int((max_length - len(indices)) * data_balance_rate)
                if n_sample > 0:
                    all_index.extend(np.random.choice(indices, n_sample))

            if keep_size:
                all_index = np.random.choice(
                    all_index, len(self.annotations), replace=False
                )
        self.index = all_index

        # Log the actual category ratio
        print("{} Dataset: {}".format(self.subset.upper(), len(self.index)))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        # header: ['# # file name', ' annotated category', ' annotated location', ' illumination']
        index = self.index[i]
        annos = self.annotations[index]
        image_path = os.path.join(self.image_root, annos[0])
        image = Image.open(image_path)
        X = self.transform(image)
        y = self.cat2id[annos[1]]

        return X, y

    def count_cats(self, indices=None):
        if indices is None:
            indices = self.index
        label_counts = {}
        for i in indices:
            label = self.annotations[i][1]
            label_counts.setdefault(label, 0)
            label_counts[label] += 1
        return label_counts


if __name__ == "__main__":
    dataset = KeyboardImageDataset("/data/keyboard_aoi_data")
    for i in range(10):
        X, y = dataset[i]
        print(X.shape, y)
