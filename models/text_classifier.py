"""
Simple LSTM text classifier for sentiment analysis.
Handles its own tokenization with a basic vocab.
"""

import logging
import re
from collections import Counter
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# default vocab for when no trained vocab is available
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def build_vocab(texts: list[str], max_size: int = 10000) -> dict[str, int]:
    """Build word->index mapping from a list of texts."""
    counter = Counter()
    for text in texts:
        tokens = tokenize(text)
        counter.update(tokens)

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, _ in counter.most_common(max_size - 2):
        vocab[word] = len(vocab)
    return vocab


def tokenize(text: str) -> list[str]:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()


def encode(text: str, vocab: dict[str, int], max_len: int = 128) -> torch.Tensor:
    """Tokenize and encode a single text string."""
    tokens = tokenize(text)
    indices = [vocab.get(t, vocab[UNK_TOKEN]) for t in tokens[:max_len]]
    # pad
    while len(indices) < max_len:
        indices.append(vocab[PAD_TOKEN])
    return torch.tensor(indices, dtype=torch.long)


class TextClassifier(nn.Module):
    """LSTM-based binary sentiment classifier."""

    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(x))
        output, (hidden, _) = self.lstm(embedded)
        # concat last hidden states from both directions
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(hidden_cat))

    @torch.no_grad()
    def predict(
        self,
        text: str,
        vocab: dict[str, int],
        labels: Optional[list[str]] = None,
    ) -> dict:
        """Classify a single text string."""
        self.eval()
        labels = labels or ["negative", "positive"]
        device = next(self.parameters()).device

        encoded = encode(text, vocab).unsqueeze(0).to(device)
        logits = self.forward(encoded)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()

        return {
            "label": labels[pred_idx] if pred_idx < len(labels) else str(pred_idx),
            "class_index": pred_idx,
            "confidence": round(probs[0][pred_idx].item(), 4),
            "probabilities": {
                labels[i] if i < len(labels) else str(i): round(probs[0][i].item(), 4)
                for i in range(self.num_classes)
            },
        }


def load_text_classifier(
    vocab_size: int = 10000,
    device: Optional[torch.device] = None,
) -> TextClassifier:
    from models.utils import get_device
    device = device or get_device()
    model = TextClassifier(vocab_size=vocab_size)
    model.to(device)
    model.eval()
    logger.info("Text classifier loaded (device=%s)", device)
    return model
