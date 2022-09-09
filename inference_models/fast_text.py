"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

from torch import nn
import torch.nn.functional as F


class FastText(nn.Module):
    """
    FastText class is the model used for the embedding.
    """
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, text):

        embedded = self.embedding(text)
        embedded = embedded.permute(1, 0, 2)
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)

        return self.fc(pooled)

    def pooling(self, text):

        embedded = self.embedding(text)
        embedded = embedded.permute(1, 0, 2)
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)

        return pooled