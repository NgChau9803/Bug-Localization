"""
Training script for BLoco Bug Localization Model.
Supports both Option A (Bi-GRU) and Option B (Code-NoN).
"""
import os
import sys
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from data.data_loader import (
    parse_bug_reports_xml,
    build_file_index,
    build_ground_truth,
    create_splits,
    read_java_file,
)
from data.vocabulary import Vocabulary, tokenize_text, tokenize_java_code
from data.dataset import BugLocalizationDataset
from models.bug_report import decompose_bug_report
from models.graph_builder import build_code_graph, graph_stats
from models.bloco import BLocoModel


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn_code_non(batch, bug_vocab, max_clue_len, max_clues):
    """Collate for Code-NoN (Option B): builds graphs from source code."""
    batch_size = len(batch)

    clue_indices = torch.zeros(batch_size, max_clues, max_clue_len, dtype=torch.long)
    labels = torch.zeros(batch_size, dtype=torch.float)
    graphs = []

    for i, item in enumerate(batch):
        # Decompose and encode bug report
        clues = decompose_bug_report(item["bug_text"], item["bug_text"])
        for j, clue in enumerate(clues[:max_clues]):
            indices = bug_vocab.encode(clue, max_clue_len, tokenize_fn=tokenize_text)
            clue_indices[i, j] = torch.tensor(indices, dtype=torch.long)

        # Build code graph
        graph = build_code_graph(item["code_text"])
        graphs.append(graph)

        labels[i] = item["label"]

    return clue_indices, graphs, labels


def collate_fn_gru(batch, bug_vocab, code_vocab, max_clue_len, max_code_len, max_clues):
    """Collate for Bi-GRU (Option A): tokenizes source code."""
    batch_size = len(batch)

    clue_indices = torch.zeros(batch_size, max_clues, max_clue_len, dtype=torch.long)
    code_indices = torch.zeros(batch_size, max_code_len, dtype=torch.long)
    labels = torch.zeros(batch_size, dtype=torch.float)

    for i, item in enumerate(batch):
        clues = decompose_bug_report(item["bug_text"], item["bug_text"])
        for j, clue in enumerate(clues[:max_clues]):
            indices = bug_vocab.encode(clue, max_clue_len, tokenize_fn=tokenize_text)
            clue_indices[i, j] = torch.tensor(indices, dtype=torch.long)

        code_idx = code_vocab.encode(item["code_text"], max_code_len, tokenize_fn=tokenize_java_code)
        code_indices[i] = torch.tensor(code_idx, dtype=torch.long)

        labels[i] = item["label"]

    return clue_indices, code_indices, labels


def build_vocabularies(bug_reports, file_index, max_vocab_size=50000, max_files=2000):
    """Build vocabularies for bug text and (optionally) code text."""
    bug_vocab = Vocabulary(max_size=max_vocab_size, min_freq=2)
    bug_texts = [br["summary"] + " " + br["description"] for br in bug_reports]
    bug_vocab.build_from_texts(bug_texts, tokenize_fn=tokenize_text)
    return bug_vocab


def build_code_vocabulary(file_index, max_vocab_size=50000, max_files=2000):
    """Build code vocabulary (only needed for Option A)."""
    code_vocab = Vocabulary(max_size=max_vocab_size, min_freq=2)
    file_paths = list(file_index.values())
    sample_paths = random.sample(file_paths, min(max_files, len(file_paths)))
    code_texts = [read_java_file(p, max_chars=30000) for p in tqdm(sample_paths, desc="Reading code for vocab")]
    code_vocab.build_from_texts(code_texts, tokenize_fn=tokenize_java_code)
    return code_vocab


def train_epoch(model, dataloader, optimizer, criterion, device, use_code_non):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_data in tqdm(dataloader, desc="Training"):
        if use_code_non:
            clue_indices, graphs, labels = batch_data
            clue_indices = clue_indices.to(device)
            labels = labels.to(device)
            scores = model(clue_indices, graphs)
        else:
            clue_indices, code_indices, labels = batch_data
            clue_indices = clue_indices.to(device)
            code_indices = code_indices.to(device)
            labels = labels.to(device)
            scores = model(clue_indices, code_indices)

        optimizer.zero_grad()
        loss = criterion(scores, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def validate(model, dataloader, criterion, device, use_code_non):
    """Validate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Validating"):
            if use_code_non:
                clue_indices, graphs, labels = batch_data
                clue_indices = clue_indices.to(device)
                labels = labels.to(device)
                scores = model(clue_indices, graphs)
            else:
                clue_indices, code_indices, labels = batch_data
                clue_indices = clue_indices.to(device)
                code_indices = code_indices.to(device)
                labels = labels.to(device)
                scores = model(clue_indices, code_indices)

            loss = criterion(scores, labels)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train BLoco Model")
    parser.add_argument("--project", type=str, default="Tomcat",
                        choices=list(config.PROJECTS.keys()))
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--embed_dim", type=int, default=config.EMBED_DIM)
    parser.add_argument("--neg_ratio", type=int, default=config.NEGATIVE_SAMPLE_RATIO)
    parser.add_argument("--max_clue_len", type=int, default=256)
    parser.add_argument("--max_code_len", type=int, default=config.MAX_SEQ_LEN)
    parser.add_argument("--max_clues", type=int, default=5)
    parser.add_argument("--max_bugs", type=int, default=None,
                        help="Limit number of bug reports (for debugging)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="code_non",
                        choices=["code_non", "gru"],
                        help="code_non=Option B (paper), gru=Option A (baseline)")
    parser.add_argument("--gnn_layers", type=int, default=3,
                        help="Number of GNN layers for AST/CFG")
    args = parser.parse_args()

    set_seed(args.seed)
    use_code_non = (args.mode == "code_non")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")
    print(f"[Train] Mode: {'Code-NoN (Option B)' if use_code_non else 'Bi-GRU (Option A)'}")

    # Load data
    proj_cfg = config.PROJECTS[args.project]
    print(f"\n{'='*60}")
    print(f"Training BLoco on {args.project}")
    print(f"{'='*60}")

    bug_reports = parse_bug_reports_xml(proj_cfg["bug_report_xml"])
    file_index = build_file_index(proj_cfg["source_dir"])
    ground_truth = build_ground_truth(bug_reports, file_index)

    if args.max_bugs:
        bug_reports = bug_reports[:args.max_bugs]

    train_bugs, val_bugs, test_bugs = create_splits(bug_reports)

    # Build vocabularies
    print("\n[Train] Building vocabularies...")
    bug_vocab = build_vocabularies(bug_reports, file_index)

    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    bug_vocab.save(os.path.join(config.MODEL_SAVE_DIR, f"{args.project}_bug_vocab.txt"))

    code_vocab = None
    if not use_code_non:
        code_vocab = build_code_vocabulary(file_index)
        code_vocab.save(os.path.join(config.MODEL_SAVE_DIR, f"{args.project}_code_vocab.txt"))

    # Create datasets
    train_dataset = BugLocalizationDataset(
        train_bugs, ground_truth, file_index, neg_ratio=args.neg_ratio
    )
    val_dataset = BugLocalizationDataset(
        val_bugs, ground_truth, file_index, neg_ratio=args.neg_ratio
    )

    # Data loaders
    if use_code_non:
        collate = lambda batch: collate_fn_code_non(
            batch, bug_vocab, args.max_clue_len, args.max_clues
        )
    else:
        collate = lambda batch: collate_fn_gru(
            batch, bug_vocab, code_vocab,
            args.max_clue_len, args.max_code_len, args.max_clues
        )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=0,
    )

    # Model
    model = BLocoModel(
        bug_vocab_size=len(bug_vocab),
        embed_dim=args.embed_dim,
        num_filters=config.TEXTCNN_NUM_FILTERS,
        kernel_sizes=config.TEXTCNN_KERNEL_SIZES,
        gnn_hidden_dim=config.CODE_HIDDEN_DIM,
        ast_gnn_layers=args.gnn_layers,
        cfg_gnn_layers=args.gnn_layers,
        ffn_hidden_dim=config.FFN_HIDDEN_DIM,
        max_clues=args.max_clues,
        use_code_non=use_code_non,
        code_vocab_size=len(code_vocab) if code_vocab else None,
        code_gru_layers=config.CODE_GRU_LAYERS,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Train] Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    suffix = "code_non" if use_code_non else "gru"
    best_model_path = os.path.join(config.MODEL_SAVE_DIR, f"{args.project}_{suffix}_best.pt")

    print(f"\n[Train] Starting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, use_code_non)
        val_loss = validate(model, val_loader, criterion, device, use_code_non)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Time: {elapsed:.1f}s")

        # Print Code-NoN graph parse stats
        if use_code_non:
            print(f"  {graph_stats.report()}")
            graph_stats.reset()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "args": vars(args),
                "mode": args.mode,
            }, best_model_path)
            print(f"  → Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOP_PATIENCE:
                print(f"\n[Train] Early stopping at epoch {epoch}")
                break

    print(f"\n[Train] Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"[Train] Best model saved to: {best_model_path}")


if __name__ == "__main__":
    main()
