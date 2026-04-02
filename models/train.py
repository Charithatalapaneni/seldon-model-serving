"""
Sample training script — trains the text classifier on synthetic data
and exports a versioned model artifact.

Usage: python -m models.train
"""

import logging
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.text_classifier import TextClassifier, build_vocab, encode
from models.utils import save_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# synthetic training data (just for demo purposes)
POSITIVE = [
    "this movie was great and I loved it",
    "amazing performance by the entire cast",
    "highly recommended, one of the best films this year",
    "the story was engaging and well written",
    "beautiful cinematography and excellent direction",
    "really enjoyed watching this, would see again",
    "fantastic plot twists and wonderful acting",
    "such a fun and entertaining experience",
    "the best thing I have seen in a long time",
    "incredible movie with a powerful message",
]

NEGATIVE = [
    "terrible movie, complete waste of time",
    "the acting was awful and the plot made no sense",
    "I fell asleep halfway through, very boring",
    "worst film I have seen this year",
    "poorly written script with bad dialogue",
    "the special effects were cheap and distracting",
    "I want my money back, total disappointment",
    "painfully slow and utterly forgettable",
    "the director clearly had no vision for this",
    "an absolute disaster from start to finish",
]


def generate_training_data(n_per_class: int = 200):
    """Augment training data by shuffling and recombining phrases."""
    texts, labels = [], []
    for _ in range(n_per_class):
        texts.append(random.choice(POSITIVE))
        labels.append(1)
        texts.append(random.choice(NEGATIVE))
        labels.append(0)
    return texts, labels


def train():
    texts, labels = generate_training_data(200)
    vocab = build_vocab(texts, max_size=5000)
    logger.info("Vocab size: %d", len(vocab))

    encoded = torch.stack([encode(t, vocab, max_len=64) for t in texts])
    targets = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(encoded, targets)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = TextClassifier(vocab_size=len(vocab), embed_dim=64, hidden_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10):
        total_loss = 0
        correct = 0
        total = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = output.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)

        acc = correct / total
        logger.info("Epoch %d — loss: %.4f, acc: %.2f%%", epoch + 1, total_loss / len(loader), acc * 100)

    # save the trained model
    save_model(
        model, "text_classifier",
        metadata={"vocab_size": len(vocab), "embed_dim": 64, "hidden_dim": 128},
    )

    # also save the vocab for inference
    import json
    vocab_path = save_model.__module__  # just get the artifacts dir
    from models.utils import ARTIFACTS_DIR
    vocab_file = ARTIFACTS_DIR / "text_classifier" / "vocab.json"
    vocab_file.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_file, "w") as f:
        json.dump(vocab, f)
    logger.info("Saved vocab to %s", vocab_file)

    # quick test
    model.eval()
    result = model.predict("this was an amazing film", vocab)
    logger.info("Test prediction: %s", result)


if __name__ == "__main__":
    train()
