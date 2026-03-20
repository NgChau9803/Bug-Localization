"""
Evaluation script for BLoco Bug Localization Model.
Supports both Option A (Bi-GRU) and Option B (Code-NoN).
"""
import os
import random
import argparse
import numpy as np
import torch
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
from models.bug_report import decompose_bug_report
from models.graph_builder import build_code_graph
from models.bloco import BLocoModel


def compute_metrics(ranked_files, ground_truth_files, top_k_values=None):
    """Compute Top-K, MRR, MAP for a single bug report."""
    if top_k_values is None:
        top_k_values = [1, 5, 10]

    metrics = {}

    for k in top_k_values:
        hit = len(set(ranked_files[:k]) & ground_truth_files) > 0
        metrics[f"top_{k}"] = 1.0 if hit else 0.0

    mrr = 0.0
    for rank, f in enumerate(ranked_files, 1):
        if f in ground_truth_files:
            mrr = 1.0 / rank
            break
    metrics["mrr"] = mrr

    num_relevant, precision_sum = 0, 0.0
    for rank, f in enumerate(ranked_files, 1):
        if f in ground_truth_files:
            num_relevant += 1
            precision_sum += num_relevant / rank
    metrics["ap"] = precision_sum / max(len(ground_truth_files), 1)

    return metrics


def evaluate_model(
    model, bug_reports, ground_truth, file_index,
    bug_vocab, device,
    code_vocab=None,
    use_code_non=True,
    max_clue_len=256, max_code_len=512,
    max_clues=5, max_candidates=500, batch_size=64,
):
    """Evaluate model: rank candidates per bug, compute metrics."""
    model.eval()

    all_metrics = {f"top_{k}": [] for k in config.TOP_K_VALUES}
    all_metrics.update({"mrr": [], "ap": []})

    all_file_paths = list(file_index.keys())

    with torch.no_grad():
        for br in tqdm(bug_reports, desc="Evaluating"):
            bug_id = br["bug_id"]
            buggy_files = set(ground_truth.get(bug_id, []))
            if not buggy_files:
                continue

            # Candidate files
            candidates = list(buggy_files)
            non_buggy = [f for f in all_file_paths if f not in buggy_files]
            n_extra = min(max_candidates - len(candidates), len(non_buggy))
            candidates.extend(random.sample(non_buggy, n_extra))

            # Encode clues (shared across candidates)
            clues = decompose_bug_report(br["summary"], br["description"])
            clue_idx = torch.zeros(1, max_clues, max_clue_len, dtype=torch.long)
            for j, clue in enumerate(clues[:max_clues]):
                indices = bug_vocab.encode(clue, max_clue_len, tokenize_fn=tokenize_text)
                clue_idx[0, j] = torch.tensor(indices, dtype=torch.long)

            # Score candidates in batches
            scores = []
            for i in range(0, len(candidates), batch_size):
                batch_cands = candidates[i:i + batch_size]
                bs = len(batch_cands)
                clue_batch = clue_idx.expand(bs, -1, -1).to(device)

                if use_code_non:
                    graphs = []
                    for fp in batch_cands:
                        abs_path = file_index.get(fp, "")
                        code = read_java_file(abs_path) if abs_path else ""
                        graphs.append(build_code_graph(code))
                    batch_scores = model(clue_batch, graphs)
                else:
                    code_batch = torch.zeros(bs, max_code_len, dtype=torch.long)
                    for k, fp in enumerate(batch_cands):
                        abs_path = file_index.get(fp, "")
                        code = read_java_file(abs_path) if abs_path else ""
                        idx = code_vocab.encode(code, max_code_len, tokenize_fn=tokenize_java_code)
                        code_batch[k] = torch.tensor(idx, dtype=torch.long)
                    code_batch = code_batch.to(device)
                    batch_scores = model(clue_batch, code_batch)

                scores.extend(batch_scores.cpu().tolist())

            ranked_indices = np.argsort(scores)[::-1]
            ranked_files = [candidates[i] for i in ranked_indices]

            m = compute_metrics(ranked_files, buggy_files, config.TOP_K_VALUES)
            for key in all_metrics:
                all_metrics[key].append(m[key])

    return {k: np.mean(v) if v else 0.0 for k, v in all_metrics.items()}


def main():
    parser = argparse.ArgumentParser(description="Evaluate BLoco Model")
    parser.add_argument("--project", type=str, default="Tomcat",
                        choices=list(config.PROJECTS.keys()))
    parser.add_argument("--mode", type=str, default="code_non",
                        choices=["code_non", "gru"])
    parser.add_argument("--max_clue_len", type=int, default=256)
    parser.add_argument("--max_code_len", type=int, default=config.MAX_SEQ_LEN)
    parser.add_argument("--max_clues", type=int, default=5)
    parser.add_argument("--max_candidates", type=int, default=500)
    parser.add_argument("--embed_dim", type=int, default=config.EMBED_DIM)
    parser.add_argument("--gnn_layers", type=int, default=3)
    args = parser.parse_args()

    use_code_non = (args.mode == "code_non")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device} | Mode: {args.mode}")

    # Load data
    proj_cfg = config.PROJECTS[args.project]
    bug_reports = parse_bug_reports_xml(proj_cfg["bug_report_xml"])
    file_index = build_file_index(proj_cfg["source_dir"])
    ground_truth = build_ground_truth(bug_reports, file_index)
    _, _, test_bugs = create_splits(bug_reports)

    # Load vocabularies
    bug_vocab = Vocabulary()
    bug_vocab.load(os.path.join(config.MODEL_SAVE_DIR, f"{args.project}_bug_vocab.txt"))

    code_vocab = None
    if not use_code_non:
        code_vocab = Vocabulary()
        code_vocab.load(os.path.join(config.MODEL_SAVE_DIR, f"{args.project}_code_vocab.txt"))

    # Load model
    suffix = "code_non" if use_code_non else "gru"
    model_path = os.path.join(config.MODEL_SAVE_DIR, f"{args.project}_{suffix}_best.pt")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

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

    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[Eval] Loaded model from epoch {checkpoint['epoch']}")

    # Evaluate
    print(f"\n{'='*60}")
    print(f"Evaluating on {args.project} ({len(test_bugs)} test bug reports)")
    print(f"{'='*60}")

    metrics = evaluate_model(
        model, test_bugs, ground_truth, file_index,
        bug_vocab, device,
        code_vocab=code_vocab,
        use_code_non=use_code_non,
        max_clue_len=args.max_clue_len,
        max_code_len=args.max_code_len,
        max_clues=args.max_clues,
        max_candidates=args.max_candidates,
    )

    print(f"\n{'='*40}")
    print(f"Results for {args.project} ({args.mode}):")
    print(f"{'='*40}")
    for k in config.TOP_K_VALUES:
        print(f"  Top-{k} Accuracy:  {metrics[f'top_{k}']:.4f}")
    print(f"  MRR:             {metrics['mrr']:.4f}")
    print(f"  MAP:             {metrics['ap']:.4f}")


if __name__ == "__main__":
    main()
