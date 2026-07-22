import os
import pickle
import random
import string
from typing import List

import cv2
import numpy as np
import pandas as pd
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler


class Flickr30(Dataset):
    def __init__(self,
                 images_folder_path: str,
                 split_csv_path: str,
                 vocab_stoi_path: str,
                 vocab_itos_path: str,
                 split: str,  # "train" | "val" | "test"
                 transform=None):

        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test', got '{split}'"

        self.images_folder_path = images_folder_path
        self.transform = transform
        self.spacy_model = spacy.load("en_core_web_sm")

        # ── Load only this split's rows ──────────────────────────────
        df = pd.read_csv(split_csv_path)
        self.labels = df[df["split"] == split].reset_index(drop=True)

        # ── Load vocab ────────────────────────────────────────────────
        with open(vocab_stoi_path, "rb") as f:
            self.vocab_stoi = pickle.load(f)
        with open(vocab_itos_path, "rb") as f:
            self.vocab_itos = pickle.load(f)

        # ── Pre-compute caption lengths for BucketSampler ──────────────
        self.caption_lengths = self._compute_lengths()

        print(f"  [{split:5s}] {len(self.labels):,} samples loaded")

    # ── Internal ──────────────────────────────────────────────────────
    def _compute_lengths(self) -> List[int]:
        lengths = []
        for caption in self.labels["comment"]:
            caption = "" if not isinstance(caption, str) else caption
            lengths.append(len(self._tokenize(caption)) + 2)  # +2 for sos/eos
        return lengths

    def _tokenize(self, caption: str) -> List[str]:
        return [f" {tok.text.lower()}" for tok in self.spacy_model(caption)]

    # ── Encode / decode ───────────────────────────────────────────────
    def encoder(self, caption: str) -> torch.Tensor:
        encoded = []
        for token in self._tokenize(caption.strip()):
            if token in self.vocab_stoi:
                encoded.append(self.vocab_stoi[token])
            else:
                for letter in token:
                    encoded.append(self.vocab_stoi.get(letter, self.vocab_stoi["<unk>"]))
        return torch.tensor(
            [self.vocab_stoi["<sos>"], self.vocab_stoi["<EN>"]] + encoded + [self.vocab_stoi["<eos>"]],
            dtype=torch.long,
        )

    def decoder(self, seq_of_indices) -> str:
        tokens = [self.vocab_itos.get(i, "<unk>") for i in seq_of_indices]
        tokens = [t for t in tokens if t not in ("<sos>", "<eos>", "<pad>", "<EN>")]
        words, current_word = [], []
        for token in tokens:
            if len(token) == 1 and token in string.ascii_lowercase + " ":
                if token == " ":
                    if current_word:
                        words.append("".join(current_word))
                        current_word = []
                else:
                    current_word.append(token)
            else:
                if current_word:
                    words.append("".join(current_word))
                    current_word = []
                words.append(token.strip())
        if current_word:
            words.append("".join(current_word))
        return " ".join(words)

    # ── PyTorch interface ────────────────────────────────────────────
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        image_id = row["image_name"]
        caption = row["comment"] if isinstance(row["comment"], str) else ""
        image = cv2.imread(os.path.join(self.images_folder_path, image_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, self.encoder(caption), image_id


class BucketSampler(Sampler):
    """Minimises padding by grouping similarly-lengthed captions into batches.

    Args:
        caption_lengths : list of caption lengths, one per dataset sample
        batch_size      : batch size used in the DataLoader
        bucket_size     : number of batches per bucket (default 100)
        shuffle         : True for train, False for val/test
    """

    def __init__(self,
                 caption_lengths: List[int],
                 batch_size: int,
                 bucket_size: int = 100,
                 shuffle: bool = True):
        super().__init__()
        self.shuffle = shuffle
        self.bucket_size = bucket_size * batch_size  # convert to samples
        self.sorted_indices = np.argsort(caption_lengths).tolist()

    def __iter__(self):
        indices = self.sorted_indices.copy()
        buckets = [
            indices[i:i + self.bucket_size]
            for i in range(0, len(indices), self.bucket_size)
        ]
        if self.shuffle:
            for bucket in buckets:
                random.shuffle(bucket)
            random.shuffle(buckets)
        return iter([idx for bucket in buckets for idx in bucket])

    def __len__(self):
        return len(self.sorted_indices)


def collate_fn(batch):
    images, captions, image_ids = zip(*batch)
    images = torch.stack(images, 0)
    captions = pad_sequence(captions, batch_first=True, padding_value=1)
    return images, captions, list(image_ids)
