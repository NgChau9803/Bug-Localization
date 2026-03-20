"""
Data Loader for BLoco Bug Localization
Parses XML bug reports and indexes Java source files.
"""
import os
import re
from lxml import etree
from collections import defaultdict


def parse_bug_reports_xml(xml_path: str) -> list[dict]:
    """
    Parse bug reports from phpMyAdmin XML export format.

    Returns:
        List of dicts with keys:
        - id, bug_id, summary, description,
        - report_time, report_timestamp,
        - status, commit, commit_timestamp,
        - files (list of changed file paths),
        - result (ranked result string)
    """
    tree = etree.parse(xml_path)
    root = tree.getroot()

    # Remove namespace prefixes for easier XPath
    nsmap = {"pma": "http://www.phpmyadmin.net/some_doc_url/"}

    bug_reports = []

    # Each <table> element is one bug report row
    for table_elem in root.iter("table"):
        record = {}
        for col in table_elem.findall("column"):
            col_name = col.get("name")
            col_text = col.text if col.text else ""
            record[col_name] = col_text

        # Parse the 'files' field: newline-separated file paths
        files_raw = record.get("files", "")
        files_list = [f.strip() for f in files_raw.split("\n") if f.strip()]

        bug_report = {
            "id": int(record.get("id", 0)),
            "bug_id": int(record.get("bug_id", 0)),
            "summary": record.get("summary", ""),
            "description": record.get("description", ""),
            "report_time": record.get("report_time", ""),
            "report_timestamp": float(record.get("report_timestamp", 0)),
            "status": record.get("status", ""),
            "commit": record.get("commit", ""),
            "commit_timestamp": float(record.get("commit_timestamp", 0)),
            "files": files_list,  # ground truth buggy files
            "result": record.get("result", ""),
        }
        bug_reports.append(bug_report)

    # Sort by report_timestamp for chronological splitting
    bug_reports.sort(key=lambda x: x["report_timestamp"])

    print(f"[DataLoader] Loaded {len(bug_reports)} bug reports from {os.path.basename(xml_path)}")
    return bug_reports


def build_file_index(source_dir: str) -> dict[str, str]:
    """
    Index all .java files in the source directory.

    Returns:
        Dict mapping relative_path -> absolute_path
        e.g. "org/aspectj/weaver/Checker.java" -> "D://.../Checker.java"
    """
    file_index = {}
    for root_dir, dirs, files in os.walk(source_dir):
        for fname in files:
            if fname.endswith(".java"):
                abs_path = os.path.join(root_dir, fname)
                rel_path = os.path.relpath(abs_path, source_dir)
                # Normalize to forward slashes
                rel_path = rel_path.replace("\\", "/")
                file_index[rel_path] = abs_path

    print(f"[DataLoader] Indexed {len(file_index)} Java files from {os.path.basename(source_dir)}")
    return file_index


def match_buggy_files(bug_files: list[str], file_index: dict[str, str]) -> list[str]:
    """
    Match bug report's changed files to actual source files in the index.

    Bug reports may use paths like:
      "org.aspectj.ajdt.core/src/org/aspectj/ajdt/internal/core/builder/AjState.java"
    We need to find the matching file in the index by suffix matching.

    Returns:
        List of matched relative paths from file_index
    """
    matched = []
    for bug_file in bug_files:
        bug_file_norm = bug_file.strip().replace("\\", "/")

        # Direct match
        if bug_file_norm in file_index:
            matched.append(bug_file_norm)
            continue

        # Suffix matching: check if any indexed file ends with the bug file path
        found = False
        for indexed_path in file_index:
            if indexed_path.endswith(bug_file_norm) or bug_file_norm.endswith(indexed_path):
                matched.append(indexed_path)
                found = True
                break

        if not found:
            # Try matching by filename only as last resort
            bug_filename = os.path.basename(bug_file_norm)
            candidates = [p for p in file_index if p.endswith("/" + bug_filename) or p == bug_filename]
            if len(candidates) == 1:
                matched.append(candidates[0])

    return matched


def build_ground_truth(
    bug_reports: list[dict],
    file_index: dict[str, str],
) -> dict[int, list[str]]:
    """
    Build ground truth mapping: bug_id -> list of matched buggy file paths.

    Returns:
        Dict[bug_id, List[matched_relative_path]]
    """
    ground_truth = {}
    total_matched = 0
    total_files = 0

    for br in bug_reports:
        matched = match_buggy_files(br["files"], file_index)
        ground_truth[br["bug_id"]] = matched
        total_matched += len(matched)
        total_files += len(br["files"])

    match_rate = total_matched / max(total_files, 1) * 100
    print(f"[DataLoader] Ground truth: {total_matched}/{total_files} files matched ({match_rate:.1f}%)")
    return ground_truth


def create_splits(
    bug_reports: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split bug reports chronologically into train/val/test sets.
    Bug reports should already be sorted by report_timestamp.
    """
    n = len(bug_reports)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = bug_reports[:train_end]
    val = bug_reports[train_end:val_end]
    test = bug_reports[val_end:]

    print(f"[DataLoader] Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def read_java_file(file_path: str, max_chars: int = 50000) -> str:
    """Read a Java source file with error handling."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(max_chars)
    except Exception as e:
        return ""


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config

    # Test with AspectJ (smallest)
    project = "Tomcat"
    proj_cfg = config.PROJECTS[project]

    bug_reports = parse_bug_reports_xml(proj_cfg["bug_report_xml"])
    file_index = build_file_index(proj_cfg["source_dir"])
    ground_truth = build_ground_truth(bug_reports, file_index)
    train, val, test = create_splits(bug_reports)

    # Show sample
    sample = bug_reports[0]
    print(f"\n--- Sample Bug Report ---")
    print(f"Bug ID: {sample['bug_id']}")
    print(f"Summary: {sample['summary'][:100]}")
    print(f"Files changed: {sample['files'][:3]}")
    print(f"Matched files: {ground_truth[sample['bug_id']][:3]}")
