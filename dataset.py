"""
dataset.py — Multi30k dataset loading, vocabulary building, and tokenization.
DA6401 Assignment 3
"""

import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from datasets import load_dataset
import spacy


# ─────────────────────────────────────────────────────────────────────
# Special token indices (fixed)
# ─────────────────────────────────────────────────────────────────────
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = ['<unk>', '<pad>', '<sos>', '<eos>']


class Multi30kDataset:
    """
    Loads the Multi30k DE→EN dataset and prepares vocabularies + tokenizers.

    Usage:
        ds = Multi30kDataset()
        ds.build_vocab(min_freq=2)
        ds.process_data()
        train_dl = ds.get_dataloader('train', batch_size=128)
    """

    def __init__(self):
        print("Loading Multi30k dataset …")
        raw = load_dataset("bentrevett/multi30k")
        self.raw = raw

        # Load spaCy models
        print("Loading spaCy models …")
        self.spacy_de = spacy.load("de_core_news_sm")
        self.spacy_en = spacy.load("en_core_web_sm")

        self.src_vocab = None   # dict: token → idx
        self.tgt_vocab = None
        self.src_itos  = None   # list: idx → token
        self.tgt_itos  = None

        self.data = {}          # split → list of (src_ids, tgt_ids)

    # ── Tokenisers ────────────────────────────────────────────────────

    def tokenize_de(self, text: str):
        return [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text: str):
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]

    # ── Vocabulary ────────────────────────────────────────────────────

    def build_vocab(self, min_freq: int = 2):
        """Build src (DE) and tgt (EN) vocabulary dicts."""
        src_counter = Counter()
        tgt_counter = Counter()

        for example in self.raw['train']:
            src_counter.update(self.tokenize_de(example['de']))
            tgt_counter.update(self.tokenize_en(example['en']))

        def _make_vocab(counter, min_freq):
            vocab = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
            for token, freq in counter.items():
                if freq >= min_freq:
                    vocab[token] = len(vocab)
            return vocab

        self.src_vocab = _make_vocab(src_counter, min_freq)
        self.tgt_vocab = _make_vocab(tgt_counter, min_freq)

        self.src_itos = {v: k for k, v in self.src_vocab.items()}
        self.tgt_itos = {v: k for k, v in self.tgt_vocab.items()}

        print(f"Source vocab size : {len(self.src_vocab)}")
        print(f"Target vocab size : {len(self.tgt_vocab)}")

    # ── Data processing ───────────────────────────────────────────────

    def _encode(self, tokens, vocab):
        return (
            [SOS_IDX]
            + [vocab.get(t, UNK_IDX) for t in tokens]
            + [EOS_IDX]
        )

    def process_data(self):
        """Tokenise every split and convert to index lists."""
        assert self.src_vocab is not None, "Call build_vocab() first."
        for split in ('train', 'validation', 'test'):
            encoded = []
            for example in self.raw[split]:
                src_ids = self._encode(self.tokenize_de(example['de']), self.src_vocab)
                tgt_ids = self._encode(self.tokenize_en(example['en']), self.tgt_vocab)
                encoded.append((src_ids, tgt_ids))
            self.data[split] = encoded
        print("Data processing complete.")

    # ── DataLoader ────────────────────────────────────────────────────

    def get_dataloader(self, split: str, batch_size: int = 128, shuffle: bool = None):
        assert split in self.data, "Call process_data() first."
        if shuffle is None:
            shuffle = (split == 'train')
        dataset = _TokenDataset(self.data[split])
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=_collate_fn,
            pin_memory=True,
        )


# ── Internal helpers ──────────────────────────────────────────────────

class _TokenDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def _collate_fn(batch):
    """Pad sequences in a batch to the same length."""
    src_batch, tgt_batch = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_batch, batch_first=True, padding_value=PAD_IDX
    )
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        tgt_batch, batch_first=True, padding_value=PAD_IDX
    )
    return src_padded, tgt_padded