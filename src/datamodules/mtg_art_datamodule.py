import pickle
from typing import Optional, Tuple
from os import path
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchtext.vocab import build_vocab_from_iterator
from torchvision.transforms import transforms
from torchtext.data.utils import get_tokenizer


def load_all(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


class MtgImageDataset(Dataset):
    def __init__(self, file, transform=None, tokenizer='basic_english'):
        self.data = list(load_all(file))
        self.transform = transform
        self.tokenizer = get_tokenizer(tokenizer)
        self.vocab = build_vocab_from_iterator(map(lambda x: self.tokenizer(x[1]), self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.vocab(self.tokenizer(label)))
        return image, label


class MtGArtDataModule(LightningDataModule):

    def __init__(self,
                 data_dir: str = "data/",
                 train_val_test_split: Tuple[int, int, int] = (22_000, 4_000, 6383),
                 batch_size: int = 64,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 image_size: int = 512
                 ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # data transformations
        self.image_transforms = transforms.Compose(
            [transforms.ToTensor(),
             transforms.RandomHorizontalFlip(),
             transforms.Resize(image_size),
             transforms.RandomCrop(image_size),
             transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:

        if not self.data_train and not self.data_val and not self.data_test:
            dataset = MtgImageDataset(path.join(self.hparams.data_dir, 'mtg.pkl'), self.image_transforms)
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42)
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


