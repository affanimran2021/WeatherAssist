"""SIGNS image loader; filename format is <integer-label>_<unique-id>.jpg.

Replace this dataset with your project's Dataset. The example does not assume
that this filename convention is appropriate for other tasks.
"""
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import utils


def make_transform(params, training=False):
    steps = [transforms.Resize((params.image_size, params.image_size))]
    if training and getattr(params, 'horizontal_flip', False):
        steps.append(transforms.RandomHorizontalFlip())
    steps.append(transforms.ToTensor())
    return transforms.Compose(steps)


class SIGNSDataset(Dataset):
    def __init__(self, data_dir, transform, num_classes):
        self.filenames = sorted(p for p in Path(data_dir).iterdir()
                                if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        if not self.filenames:
            raise ValueError(f'No images found in {data_dir}')
        self.labels = [int(p.name.split('_', 1)[0]) for p in self.filenames]
        if any(label < 0 or label >= num_classes for label in self.labels):
            raise ValueError('Image label is outside the configured class range.')
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        with Image.open(self.filenames[index]) as image:
            tensor = self.transform(image.convert('RGB'))
        return tensor, self.labels[index]


def fetch_dataloader(types, data_dir, params):
    loaders = {}
    for split in types:
        if split not in {'train', 'val', 'test'}:
            raise ValueError(f'Unknown split: {split}')
        dataset = SIGNSDataset(Path(data_dir) / f'{split}_signs',
                               make_transform(params, split == 'train'), params.num_classes)
        generator = torch.Generator().manual_seed(params.seed)
        loaders[split] = DataLoader(dataset, batch_size=params.batch_size,
            shuffle=split == 'train', num_workers=params.num_workers,
            pin_memory=params.device.startswith('cuda'), drop_last=False,
            worker_init_fn=utils.seed_worker, generator=generator)
    return loaders
