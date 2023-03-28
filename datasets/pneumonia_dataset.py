import os
import torch

from typing import Tuple

from torch.utils.data import Dataset
from torchvision.io import read_image


#Maybe try normalisation
#try augmentation
#try sampler that evens out classes in batches
class PneumoniaDataset(Dataset):
    """
    Dataset for accessing data for the pneumonia prediction part of the project
    
    The dataset assumes the following directory structure:
    split(e.g. train)
    |-NORMAL
    | |-img1.jpeg
    | ...
    |-PNEUMONIA
    | |-img1.jpeg
    | ...
    The label 0 corresponds to normal image, whereas the label 1 corresponds to
    image with pneumonia.
    """
    def __init__(self, img_dir: str, transform: torch.nn.Module = None, label_transform: torch.nn.Module = None, keep_in_memory: bool = True):
        self.img_dir = img_dir
        self.transform = transform
        self.label_transform = label_transform
        self.keep_in_memory = keep_in_memory

        normal_image_names = list(os.listdir(os.path.join(img_dir, "NORMAL")))
        normal_image_paths = [os.path.join(img_dir, "NORMAL", name) for name in normal_image_names]
        pneumonia_image_names = list(os.listdir(os.path.join(img_dir, "PNEUMONIA")))
        pneumonia_image_paths = [os.path.join(img_dir, "PNEUMONIA", name) for name in pneumonia_image_names]
        self.image_paths = normal_image_paths + pneumonia_image_paths
        self.labels = [0]*len(normal_image_paths) + [1]*len(pneumonia_image_paths)

        if self.keep_in_memory:
            self.images = []
            for path in self.image_paths:
                image = read_image(path) / 255.
                self.images.append(image)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        if self.keep_in_memory:
            image = self.images[idx] / 255.
        else:
            image = read_image(os.path.join(self.img_dir, self.image_paths[idx])) / 255.

        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.label_transform:
            label = self.label_transform(label)
        return image.repeat(3,1,1), label