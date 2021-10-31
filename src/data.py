import os
import torchvision
import torch
import pandas as pd

def read_data_my(data_dir, is_train=True):
    csv_fname = os.path.join(data_dir,
                             'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(
            torchvision.io.read_image(
                os.path.join(data_dir,
                             'images', f'{img_name}')))
        # Here `target` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
        targets.append(list(target))
    targets = torch.tensor(targets).unsqueeze(1)
    # targets[:, :, 1:] = targets[:, :, 1:] / 256.
    targets = targets.float()
    targets[:, :, 1:] = targets[:, :, 1:].float() / 256.
    # targets[:, :, 1:] = torch.div(targets[:, :, 1:].float(), 0.137)

    return images, targets

class MyDataset(torch.utils.data.Dataset):
    """A customized dataset to load the banana detection dataset."""
    def __init__(self, data_dir, is_train):
        self.features, self.labels = read_data_my(data_dir, is_train)
        print('read ' + str(len(self.features)) + (
            f' training examples' if is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)