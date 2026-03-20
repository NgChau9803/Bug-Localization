"""
Run full experiments: Train + Evaluate on all projects.
Outputs comparison table with paper baselines.

Usage:
    .venv\Scripts\activate
    python run_experiments.py
    python run_experiments.py --project Tomcat   (single project)
    python run_experiments.py --mode gru          (Option A for comparison)
"""
import os
import sys
import json
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from data.data_loader import (
    parse_bug_reports_xml, build_file_index,
    build_ground_truth, create_splits, read_java_file,
)
from data.vocabulary import Vocabulary, tokenize_text, tokenize_java_code
from data.dataset import BugLocalizationDataset
from models.bug_report import decompose_bug_report
from models.graph_builder import build_code_graph, graph_stats
from models.bloco import BLocoModel
from evaluate import evaluate_model, compute_metrics


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn_code_non(batch, bug_vocab, max_clue_len, max_clues):
    bs = len(batch)
    clue_indices = torch.zeros(bs, max_clues, max_clue_len, dtype=torch.long)
    labels = torch.zeros(bs, dtype=torch.float)
    graphs = []
    for i, item in enumerate(batch):
        clues = decompose_bug_report(item["bug_text"], item["bug_text"])
        for j, clue in enumerate(clues[:max_clues]):
            idx = bug_vocab.encode(clue, max_clue_len, tokenize_fn=tokenize_text)
            clue_indices[i, j] = torch.tensor(idx, dtype=torch.long)
        graphs.append(build_code_graph(item["code_text"]))
        labels[i] = item["label"]
    return clue_indices, graphs, labels


def collate_fn_gru(batch, bug_vocab, code_vocab, max_clue_len, max_code_len, max_clues):
    bs = len(batch)
    clue_indices = torch.zeros(bs, max_clues, max_clue_len, dtype=torch.long)
    code_indices = torch.zeros(bs, max_code_len, dtype=torch.long)
    labels = torch.zeros(bs, dtype=torch.float)
    for i, item in enumerate(batch):
        clues = decompose_bug_report(item["bug_text"], item["bug_text"])
        for j, clue in enumerate(clues[:max_clues]):
            idx = bug_vocab.encode(clue, max_clue_len, tokenize_fn=tokenize_text)
            clue_indices[i, j] = torch.tensor(idx, dtype=torch.long)
        ci = code_vocab.encode(item["code_text"], max_code_len, tokenize_fn=tokenize_java_code)
        code_indices[i] = torch.tensor(ci, dtype=torch.long)
        labels[i] = item["label"]
    return clue_indices, code_indices, labels


def run_project(project_name, mode="code_non", epochs=30, batch_size=16,
                lr=0.001, neg_ratio=5, embed_dim=200, gnn_layers=3,
                max_clue_len=256, max_code_len=512, max_clues=5,
                max_candidates=300, seed=42):
    """Train and evaluate on a single project."""
    set_seed(seed)
    use_code_non = (mode == "code_non")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    proj_cfg = config.PROJECTS[project_name]
    print(f"\n{'='*70}")
    print(f"  {project_name} | Mode: {'Code-NoN' if use_code_non else 'Bi-GRU'} | Device: {device}")
    print(f"{'='*70}")

    # Load data
    bug_reports = parse_bug_reports_xml(proj_cfg["bug_report_xml"])
    file_index = build_file_index(proj_cfg["source_dir"])
    ground_truth = build_ground_truth(bug_reports, file_index)
    train_bugs, val_bugs, test_bugs = create_splits(bug_reports)

    # Vocabularies
    bug_vocab = Vocabulary(max_size=50000, min_freq=2)
    bug_texts = [br["summary"] + " " + br["description"] for br in bug_reports]
    bug_vocab.build_from_texts(bug_texts, tokenize_fn=tokenize_text)

    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    bug_vocab.save(os.path.join(config.MODEL_SAVE_DIR, f"{project_name}_bug_vocab.txt"))

    code_vocab = None
    if not use_code_non:
        code_vocab = Vocabulary(max_size=50000, min_freq=2)
        paths = list(file_index.values())
        sample = random.sample(paths, min(2000, len(paths)))
        texts = [read_java_file(p, 30000) for p in tqdm(sample, desc="Code vocab")]
        code_vocab.build_from_texts(texts, tokenize_fn=tokenize_java_code)
        code_vocab.save(os.path.join(config.MODEL_SAVE_DIR, f"{project_name}_code_vocab.txt"))

    # Datasets
    train_ds = BugLocalizationDataset(train_bugs, ground_truth, file_index, neg_ratio=neg_ratio)
    val_ds = BugLocalizationDataset(val_bugs, ground_truth, file_index, neg_ratio=neg_ratio)

    if use_code_non:
        collate = lambda b: collate_fn_code_non(b, bug_vocab, max_clue_len, max_clues)
    else:
        collate = lambda b: collate_fn_gru(b, bug_vocab, code_vocab, max_clue_len, max_code_len, max_clues)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    # Model
    model = BLocoModel(
        bug_vocab_size=len(bug_vocab),
        embed_dim=embed_dim,
        num_filters=config.TEXTCNN_NUM_FILTERS,
        kernel_sizes=config.TEXTCNN_KERNEL_SIZES,
        gnn_hidden_dim=config.CODE_HIDDEN_DIM,
        ast_gnn_layers=gnn_layers,
        cfg_gnn_layers=gnn_layers,
        ffn_hidden_dim=config.FFN_HIDDEN_DIM,
        max_clues=max_clues,
        use_code_non=use_code_non,
        code_vocab_size=len(code_vocab) if code_vocab else None,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    suffix = "code_non" if use_code_non else "gru"
    model_path = os.path.join(config.MODEL_SAVE_DIR, f"{project_name}_{suffix}_best.pt")
    best_val_loss = float("inf")
    patience = 0

    # ========== TRAIN ==========
    start_all = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_batch = 0, 0

        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            if use_code_non:
                clue_idx, graphs, labels = batch_data
                clue_idx, labels = clue_idx.to(device), labels.to(device)
                scores = model(clue_idx, graphs)
            else:
                clue_idx, code_idx, labels = batch_data
                clue_idx = clue_idx.to(device)
                code_idx, labels = code_idx.to(device), labels.to(device)
                scores = model(clue_idx, code_idx)

            optimizer.zero_grad()
            loss = criterion(scores, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1

        train_loss = total_loss / max(n_batch, 1)

        # Validate
        model.eval()
        val_loss_total, val_n = 0, 0
        with torch.no_grad():
            for batch_data in val_loader:
                if use_code_non:
                    clue_idx, graphs, labels = batch_data
                    clue_idx, labels = clue_idx.to(device), labels.to(device)
                    scores = model(clue_idx, graphs)
                else:
                    clue_idx, code_idx, labels = batch_data
                    clue_idx = clue_idx.to(device)
                    code_idx, labels = code_idx.to(device), labels.to(device)
                    scores = model(clue_idx, code_idx)
                val_loss_total += criterion(scores, labels).item()
                val_n += 1

        val_loss = val_loss_total / max(val_n, 1)

        # Graph stats
        graph_info = ""
        if use_code_non:
            graph_info = f" | {graph_stats.report()}"
            graph_stats.reset()

        print(f"  Epoch {epoch:2d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}{graph_info}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_loss": val_loss, "mode": mode,
            }, model_path)
        else:
            patience += 1
            if patience >= config.EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    train_time = time.time() - start_all
    print(f"  Training time: {train_time:.0f}s | Best val loss: {best_val_loss:.4f}")

    # ========== EVALUATE ==========
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"\n  Evaluating on {len(test_bugs)} test bug reports...")

    metrics = evaluate_model(
        model, test_bugs, ground_truth, file_index,
        bug_vocab, device,
        code_vocab=code_vocab,
        use_code_non=use_code_non,
        max_clue_len=max_clue_len,
        max_code_len=max_code_len,
        max_clues=max_clues,
        max_candidates=max_candidates,
    )

    print(f"\n  {'='*40}")
    print(f"  Results for {project_name} ({mode}):")
    for k in config.TOP_K_VALUES:
        print(f"    Top-{k}:  {metrics[f'top_{k}']:.4f}")
    print(f"    MRR:   {metrics['mrr']:.4f}")
    print(f"    MAP:   {metrics['ap']:.4f}")
    print(f"  {'='*40}")

    return {
        "project": project_name,
        "mode": mode,
        "train_time": train_time,
        "best_val_loss": best_val_loss,
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default=None,
                        help="Single project, or None for all")
    parser.add_argument("--mode", type=str, default="code_non",
                        choices=["code_non", "gru"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--neg_ratio", type=int, default=5)
    parser.add_argument("--max_candidates", type=int, default=300)
    args = parser.parse_args()

    projects = [args.project] if args.project else list(config.PROJECTS.keys())

    all_results = []
    for proj in projects:
        result = run_project(
            proj, mode=args.mode, epochs=args.epochs,
            batch_size=args.batch_size, neg_ratio=args.neg_ratio,
            max_candidates=args.max_candidates,
        )
        all_results.append(result)

    # Save results
    results_path = os.path.join(config.MODEL_SAVE_DIR, f"results_{args.mode}.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print comparison table
    print(f"\n\n{'='*80}")
    print(f"  FINAL RESULTS — BLoco ({'Code-NoN' if args.mode == 'code_non' else 'Bi-GRU'})")
    print(f"{'='*80}")
    print(f"  {'Project':<20} {'Top-1':>8} {'Top-5':>8} {'Top-10':>8} {'MRR':>8} {'MAP':>8}")
    print(f"  {'-'*60}")
    for r in all_results:
        print(f"  {r['project']:<20} "
              f"{r['top_1']:>8.4f} {r['top_5']:>8.4f} {r['top_10']:>8.4f} "
              f"{r['mrr']:>8.4f} {r['ap']:>8.4f}")
    print(f"\n  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
