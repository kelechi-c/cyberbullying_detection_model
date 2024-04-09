import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import pytorch_lightning as pl

class TweetDataset(pl.LightningDataModule):
    def __init__(self, csv_file, batch_size=32):
        super().__init__()
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = None
        self.input_size = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def setup(self, stage=None):
        # Load the data
        data = pd.read_csv(self.csv_file)

        # Split the data into train, val, and test sets
        self.train_data, self.val_data = train_test_split(data, test_size=0.2, random_state=42)
        self.test_data = self.val_data

        # Build the vocabulary
        text_iterator = (text for text in self.train_data['text'])
        self.vocab = build_vocab_from_iterator(text_iterator, specials=['<UNK>'])
        self.vocab.set_default_index(self.vocab['<UNK>'])

        # Define the input size based on the vocabulary size
        self.input_size = len(self.vocab)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        text = [self.vocab(self.tokenizer(row['text'])) for row in batch]
        label = [row['label'] for row in batch]
        return torch.tensor(text), torch.tensor(label)

    def __len__(self):
        if self.train_data is not None:
            return len(self.train_data)
        elif self.val_data is not None:
            return len(self.val_data)
        elif self.test_data is not None:
            return len(self.test_data)
        else:
            return 0