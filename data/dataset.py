"""
PyTorch Dataset for BLoco Bug Localization.
"""
import os
import sys
import random
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_loader import read_java_file


class BugLocalizationDataset(Dataset):
    """
    Dataset for bug localization.

    For each bug report, produces pairs of (bug_report, source_file, label).
    - Positive pairs: bug report + each ground truth buggy file (label=1)
    - Negative pairs: bug report + randomly sampled non-buggy files (label=0)

    Args:
        bug_reports: list of bug report dicts
        ground_truth: dict mapping bug_id -> list of buggy file relative paths
        file_index: dict mapping relative_path -> absolute_path
        neg_ratio: number of negative samples per positive sample
        max_code_chars: max characters to read from each source file
    """

    def __init__(
        self,
        bug_reports: list[dict],
        ground_truth: dict[int, list[str]],
        file_index: dict[str, str],
        neg_ratio: int = 10,
        max_code_chars: int = 50000,
    ):
        self.bug_reports = bug_reports
        self.ground_truth = ground_truth
        self.file_index = file_index
        self.neg_ratio = neg_ratio
        self.max_code_chars = max_code_chars

        # All available file paths (for negative sampling)
        self.all_file_paths = list(file_index.keys())

        # Build pairs: (bug_report_idx, file_rel_path, label)
        self.pairs = self._build_pairs()
        print(f"[Dataset] Built {len(self.pairs)} pairs "
              f"({sum(1 for _, _, l in self.pairs if l == 1)} pos, "
              f"{sum(1 for _, _, l in self.pairs if l == 0)} neg)")

    def _build_pairs(self) -> list[tuple[int, str, int]]:
        """Build positive and negative pairs."""
        pairs = []

        for idx, br in enumerate(self.bug_reports):
            bug_id = br["bug_id"]
            buggy_files = self.ground_truth.get(bug_id, [])

            if not buggy_files:
                continue

            # Positive pairs
            for bf in buggy_files:
                pairs.append((idx, bf, 1))

            # Negative pairs: sample non-buggy files
            buggy_set = set(buggy_files)
            non_buggy = [f for f in self.all_file_paths if f not in buggy_set]

            n_neg = min(len(buggy_files) * self.neg_ratio, len(non_buggy))
            neg_samples = random.sample(non_buggy, n_neg)

            for nf in neg_samples:
                pairs.append((idx, nf, 0))

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        br_idx, file_path, label = self.pairs[idx]
        br = self.bug_reports[br_idx]

        # Bug report text (summary + description)
        bug_text = br["summary"] + " " + br["description"]

        # Source code text
        abs_path = self.file_index.get(file_path, "")
        code_text = read_java_file(abs_path, self.max_code_chars) if abs_path else ""

        return {
            "bug_text": bug_text,
            "code_text": code_text,
            "label": label,
            "bug_id": br["bug_id"],
            "file_path": file_path,
        }


class RankingDataset(Dataset):
    """
    Dataset for evaluation: for each bug report, return ALL candidate files
    so we can rank them. Used during evaluation only.

    Returns one item per bug report.
    """

    def __init__(
        self,
        bug_reports: list[dict],
        ground_truth: dict[int, list[str]],
        file_index: dict[str, str],
        max_code_chars: int = 50000,
        max_candidates: int = 500,
    ):
        self.bug_reports = [
            br for br in bug_reports
            if len(ground_truth.get(br["bug_id"], [])) > 0
        ]
        self.ground_truth = ground_truth
        self.file_index = file_index
        self.max_code_chars = max_code_chars
        self.max_candidates = max_candidates
        self.all_file_paths = list(file_index.keys())

        print(f"[RankingDataset] {len(self.bug_reports)} bug reports with ground truth")

    def __len__(self) -> int:
        return len(self.bug_reports)

    def __getitem__(self, idx: int) -> dict:
        br = self.bug_reports[idx]
        bug_id = br["bug_id"]
        bug_text = br["summary"] + " " + br["description"]
        buggy_files = set(self.ground_truth.get(bug_id, []))

        # Candidate files: include all buggy + sample of non-buggy
        candidates = list(buggy_files)
        non_buggy = [f for f in self.all_file_paths if f not in buggy_files]

        n_extra = min(self.max_candidates - len(candidates), len(non_buggy))
        candidates.extend(random.sample(non_buggy, n_extra))
        random.shuffle(candidates)

        # Labels
        labels = [1 if f in buggy_files else 0 for f in candidates]

        # Read code
        code_texts = []
        for fp in candidates:
            abs_path = self.file_index.get(fp, "")
            code = read_java_file(abs_path, self.max_code_chars) if abs_path else ""
            code_texts.append(code)

        return {
            "bug_text": bug_text,
            "bug_id": bug_id,
            "candidates": candidates,
            "code_texts": code_texts,
            "labels": labels,
        }


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    import config
    from data.data_loader import (
        parse_bug_reports_xml,
        build_file_index,
        build_ground_truth,
        create_splits,
    )

    project = "Tomcat"
    proj_cfg = config.PROJECTS[project]

    bug_reports = parse_bug_reports_xml(proj_cfg["bug_report_xml"])
    file_index = build_file_index(proj_cfg["source_dir"])
    ground_truth = build_ground_truth(bug_reports, file_index)
    train, val, test = create_splits(bug_reports)

    dataset = BugLocalizationDataset(train, ground_truth, file_index, neg_ratio=5)
    sample = dataset[0]
    print(f"\n--- Sample ---")
    print(f"Bug ID: {sample['bug_id']}")
    print(f"Bug text: {sample['bug_text'][:100]}...")
    print(f"Code text: {sample['code_text'][:100]}...")
    print(f"Label: {sample['label']}")
    print(f"File: {sample['file_path']}")
