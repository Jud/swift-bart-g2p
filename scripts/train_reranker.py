#!/usr/bin/env python3
"""Train a 1D CNN reranker for G2P candidate rescoring.

Usage:
    python3 scripts/train_reranker.py data/reranker_train.tsv Sources/BARTG2P/Resources/reranker.safetensors

Input TSV columns: word, candidate, model_logprob, trigram_logprob, label
"""

import argparse
import json
import struct
import sys
from collections import defaultdict

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Unified vocab: 4 special + 79 unique chars (sorted by codepoint)
SPECIAL = ["\x00", "\x01", "\x02", "\x03"]  # PAD, BOS, SEP, EOS
CHARS = list(
    "'-."
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "æðŋθɑɔəɛɜɡɪɹɾʃʊʌʒʔʤʧˈˌᵊᵻ"
)
VOCAB = SPECIAL + CHARS
CHAR2ID = {c: i for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)  # 83

PAD, BOS, SEP, EOS = 0, 1, 2, 3
TYPE_GRAPHEME, TYPE_PHONEME, TYPE_SPECIAL = 0, 1, 2


def tokenize(word: str, candidate: str, max_len: int = 128):
    """Encode (word, candidate) into token_ids and type_ids."""
    token_ids = [BOS]
    type_ids = [TYPE_SPECIAL]

    for c in word.lower():
        token_ids.append(CHAR2ID.get(c, PAD))
        type_ids.append(TYPE_GRAPHEME)

    token_ids.append(SEP)
    type_ids.append(TYPE_SPECIAL)

    for c in candidate:
        token_ids.append(CHAR2ID.get(c, PAD))
        type_ids.append(TYPE_PHONEME)

    token_ids.append(EOS)
    type_ids.append(TYPE_SPECIAL)

    # Truncate
    token_ids = token_ids[:max_len]
    type_ids = type_ids[:max_len]
    return token_ids, type_ids


def load_data(path: str):
    """Load TSV training data, grouped by word."""
    word_groups = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 5:
                continue
            word, candidate, model_lp, trigram_lp, label = parts
            word_groups[word].append({
                "candidate": candidate,
                "model_lp": float(model_lp),
                "trigram_lp": float(trigram_lp),
                "label": int(label),
            })
    return word_groups


def save_safetensors(tensors: dict, path: str):
    """Save float32 tensors to safetensors format."""
    header = {}
    offset = 0
    tensor_data = []
    for name, arr in tensors.items():
        arr = arr.astype(np.float32)
        data = arr.tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        tensor_data.append(data)
        offset += len(data)

    header_json = json.dumps(header).encode("utf-8")
    # Pad header to 8-byte alignment
    pad_len = (8 - len(header_json) % 8) % 8
    header_json += b" " * pad_len

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for d in tensor_data:
            f.write(d)


if not HAS_TORCH:
    print("ERROR: PyTorch is required. Install with: pip install torch", file=sys.stderr)
    sys.exit(1)


class RerankerModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=32, num_types=3, dropout=0.0):
        super().__init__()
        self.char_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.type_embed = nn.Embedding(num_types, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(66, 32)
        self.fc2 = nn.Linear(32, 1)

        # Feature normalization (set from training data statistics)
        self.register_buffer("feat_mean", torch.zeros(2))
        self.register_buffer("feat_std", torch.ones(2))

    def set_feat_stats(self, model_lp_mean, model_lp_std, trigram_lp_mean, trigram_lp_std):
        self.feat_mean[0] = model_lp_mean
        self.feat_mean[1] = trigram_lp_mean
        self.feat_std[0] = model_lp_std
        self.feat_std[1] = trigram_lp_std

    def forward(self, token_ids, type_ids, model_lp, trigram_lp):
        # token_ids, type_ids: [B, L]
        # model_lp, trigram_lp: [B]
        x = self.char_embed(token_ids) + self.type_embed(type_ids)  # [B, L, 32]
        x = x.transpose(1, 2)  # [B, 32, L]
        x = self.drop(F.relu(self.bn1(self.conv1(x))))
        x = self.drop(F.relu(self.bn2(self.conv2(x))))
        x = x.mean(dim=2)  # [B, 64]
        # Normalize logprob features
        norm_mlp = (model_lp - self.feat_mean[0]) / self.feat_std[0]
        norm_tlp = (trigram_lp - self.feat_mean[1]) / self.feat_std[1]
        features = torch.cat([x, norm_mlp.unsqueeze(1), norm_tlp.unsqueeze(1)], dim=1)
        h = self.drop(F.relu(self.fc1(features)))
        return self.fc2(h).squeeze(1)  # [B]


class RerankerDataset(Dataset):
    """Dataset of word groups for listwise ranking."""

    def __init__(self, word_groups: dict, max_len: int = 128):
        self.groups = []
        for word, candidates in word_groups.items():
            has_pos = any(c["label"] == 1 for c in candidates)
            has_neg = any(c["label"] == 0 for c in candidates)
            if not (has_pos and has_neg):
                continue

            entries = []
            for c in candidates:
                tids, tyids = tokenize(word, c["candidate"], max_len)
                entries.append({
                    "token_ids": tids,
                    "type_ids": tyids,
                    "model_lp": c["model_lp"],
                    "trigram_lp": c["trigram_lp"],
                    "label": c["label"],
                })
            self.groups.append(entries)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        return self.groups[idx]


def collate_groups(batch):
    """Collate a batch of word groups into padded tensors."""
    all_tids = []
    all_tyids = []
    all_mlp = []
    all_tlp = []
    all_labels = []
    group_sizes = []

    for group in batch:
        group_sizes.append(len(group))
        for entry in group:
            all_tids.append(entry["token_ids"])
            all_tyids.append(entry["type_ids"])
            all_mlp.append(entry["model_lp"])
            all_tlp.append(entry["trigram_lp"])
            all_labels.append(entry["label"])

    # Pad sequences
    max_len = max(len(t) for t in all_tids)
    padded_tids = [t + [PAD] * (max_len - len(t)) for t in all_tids]
    padded_tyids = [t + [TYPE_SPECIAL] * (max_len - len(t)) for t in all_tyids]

    return {
        "token_ids": torch.tensor(padded_tids, dtype=torch.long),
        "type_ids": torch.tensor(padded_tyids, dtype=torch.long),
        "model_lp": torch.tensor(all_mlp, dtype=torch.float32),
        "trigram_lp": torch.tensor(all_tlp, dtype=torch.float32),
        "labels": torch.tensor(all_labels, dtype=torch.float32),
        "group_sizes": group_sizes,
    }


def ranking_loss(scores, labels, group_sizes):
    """ListMLE ranking loss over groups of candidates."""
    loss = torch.tensor(0.0, device=scores.device)
    offset = 0
    n_groups = 0

    for size in group_sizes:
        g_scores = scores[offset:offset + size]
        g_labels = labels[offset:offset + size]

        # Sort by label descending (correct candidates first)
        _, sorted_idx = g_labels.sort(descending=True)
        sorted_scores = g_scores[sorted_idx]

        # ListMLE: product of conditional softmax probabilities
        for i in range(size - 1):
            remaining = sorted_scores[i:]
            loss -= remaining[0] - torch.logsumexp(remaining, dim=0)

        offset += size
        n_groups += 1

    return loss / max(n_groups, 1)


def evaluate(model, dataloader, device):
    """Evaluate ranking accuracy: fraction of groups where correct candidate ranks first."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            scores = model(
                batch["token_ids"].to(device),
                batch["type_ids"].to(device),
                batch["model_lp"].to(device),
                batch["trigram_lp"].to(device),
            )
            offset = 0
            for size in batch["group_sizes"]:
                g_scores = scores[offset:offset + size].cpu()
                g_labels = batch["labels"][offset:offset + size]
                best_idx = g_scores.argmax().item()
                if g_labels[best_idx] == 1:
                    correct += 1
                total += 1
                offset += size

    return correct / max(total, 1)


def export_model(model, path):
    """Export model weights to safetensors format."""
    model.eval()
    sd = model.state_dict()
    tensors = {}
    for name, param in sd.items():
        if "num_batches_tracked" in name:
            continue
        tensors[name] = param.cpu().numpy()

    save_safetensors(tensors, path)
    total_params = sum(v.size for v in tensors.values())
    size_kb = sum(v.nbytes for v in tensors.values()) / 1024
    print(f"Exported {len(tensors)} tensors, {total_params} params, {size_kb:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description="Train G2P reranker")
    parser.add_argument("input_tsv", help="Training data TSV")
    parser.add_argument("output_safetensors", help="Output weights path")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

    # Load and split data
    print("Loading data...")
    word_groups = load_data(args.input_tsv)
    words = sorted(word_groups.keys())
    np.random.shuffle(words)

    n = len(words)
    n_train = int(n * 0.90)
    n_val = int(n * 0.05)

    train_words = {w: word_groups[w] for w in words[:n_train]}
    val_words = {w: word_groups[w] for w in words[n_train:n_train + n_val]}
    test_words = {w: word_groups[w] for w in words[n_train + n_val:]}

    print(f"Split: {len(train_words)} train, {len(val_words)} val, {len(test_words)} test")

    train_ds = RerankerDataset(train_words)
    val_ds = RerankerDataset(val_words)
    test_ds = RerankerDataset(test_words)

    print(f"Groups with pos+neg: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")

    # Compute feature normalization stats from training data
    all_model_lp = []
    all_trigram_lp = []
    for group in train_ds.groups:
        for entry in group:
            all_model_lp.append(entry["model_lp"])
            all_trigram_lp.append(entry["trigram_lp"])
    mlp_mean, mlp_std = np.mean(all_model_lp), max(np.std(all_model_lp), 1e-6)
    tlp_mean, tlp_std = np.mean(all_trigram_lp), max(np.std(all_trigram_lp), 1e-6)
    print(f"Feature stats: model_lp mean={mlp_mean:.2f} std={mlp_std:.2f}, "
          f"trigram_lp mean={tlp_mean:.2f} std={tlp_std:.2f}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate_groups, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_groups, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         collate_fn=collate_groups, num_workers=0)

    # Model
    model = RerankerModel().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model params: {param_count}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2)

    best_val_acc = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_dl:
            scores = model(
                batch["token_ids"].to(device),
                batch["type_ids"].to(device),
                batch["model_lp"].to(device),
                batch["trigram_lp"].to(device),
            )
            loss = ranking_loss(scores, batch["labels"].to(device), batch["group_sizes"])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        val_acc = evaluate(model, val_dl, device)
        scheduler.step(val_acc)
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d}: loss={avg_loss:.4f}  val_acc={val_acc:.4f}  lr={lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    test_acc = evaluate(model, test_dl, device)
    print(f"\nBest val_acc: {best_val_acc:.4f}")
    print(f"Test acc: {test_acc:.4f}")

    # Export
    export_model(model, args.output_safetensors)
    print(f"Saved to {args.output_safetensors}")


if __name__ == "__main__":
    main()
