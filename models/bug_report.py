"""
Bug Report Processing: Decomposition + TextCNN Encoder
Following BLoco paper Section 2.
"""
import re
import torch
import torch.nn as nn
import torch.nn.functional as F


def decompose_bug_report(summary: str, description: str) -> list[str]:
    """
    Decompose a bug report into multiple clues.
    Following the paper: summary, description, stack trace, expected behavior,
    code snippets / patches.

    Returns:
        List of clue strings (at least 1, up to 5).
    """
    clues = []

    # Clue 1: Summary (always present)
    if summary.strip():
        clues.append(summary.strip())

    # Clue 2: General description (remove stack traces and code blocks)
    desc = description if description else ""

    # Clue 3: Stack trace
    stack_trace = extract_stack_trace(desc)
    if stack_trace:
        clues.append(stack_trace)

    # Clue 4: Expected / observed behavior
    expected = extract_expected_behavior(desc)
    if expected:
        clues.append(expected)

    # Clue 5: Code snippets / patches
    code_snippets = extract_code_snippets(desc)
    if code_snippets:
        clues.append(code_snippets)

    # Clean description: remove extracted parts
    clean_desc = desc
    if stack_trace:
        clean_desc = clean_desc.replace(stack_trace, "")
    if expected:
        clean_desc = clean_desc.replace(expected, "")
    if code_snippets:
        clean_desc = clean_desc.replace(code_snippets, "")
    clean_desc = clean_desc.strip()

    if clean_desc:
        clues.insert(1, clean_desc)  # Insert after summary

    # Ensure at least one clue
    if not clues:
        clues.append(summary if summary else "empty bug report")

    return clues


def extract_stack_trace(text: str) -> str:
    """Extract Java stack trace from text."""
    # Match patterns like "at org.xxx.Yyy.method(File.java:123)"
    stack_pattern = r'(?:(?:Exception|Error|Caused by|at ).*\n?)+'
    matches = re.findall(stack_pattern, text)
    if matches:
        # Return the longest match (most complete stack trace)
        return max(matches, key=len).strip()
    return ""


def extract_expected_behavior(text: str) -> str:
    """Extract expected/observed behavior sections."""
    patterns = [
        r'(?:expected|actual|observed|should|instead)[:\s].*?(?=\n\n|\Z)',
        r'(?:steps to reproduce)[:\s].*?(?=\n\n|\Z)',
    ]
    results = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        results.extend(matches)
    return "\n".join(results).strip()


def extract_code_snippets(text: str) -> str:
    """Extract code snippets and patches from text."""
    # Match indented code blocks or diff-like patches
    patterns = [
        r'```.*?```',               # Markdown code blocks
        r'(?:^[ \t]{4,}.*$\n?)+',   # Indented code blocks
        r'(?:^[+-].*$\n?){3,}',      # Diff-like patches
    ]
    results = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
        results.extend(matches)
    return "\n".join(results).strip()


class TextCNN(nn.Module):
    """
    TextCNN encoder for bug report clues.
    Following the paper: Conv1D with multiple kernel sizes + max pooling.

    Args:
        vocab_size: vocabulary size
        embed_dim: embedding dimension
        num_filters: number of filters per kernel size
        kernel_sizes: list of kernel sizes
        output_dim: output embedding dimension
        dropout: dropout rate
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 200,
        num_filters: int = 100,
        kernel_sizes: list[int] = None,
        output_dim: int = 200,
        dropout: float = 0.3,
    ):
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [2, 3, 4, 5]

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, ks)
            for ks in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)

        total_filters = num_filters * len(kernel_sizes)
        self.fc = nn.Linear(total_filters, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) token indices

        Returns:
            (batch, output_dim) clue embedding
        """
        # (batch, seq_len, embed_dim)
        embedded = self.embedding(x)
        # (batch, embed_dim, seq_len) for Conv1d
        embedded = embedded.permute(0, 2, 1)

        # Apply convolutions + ReLU + max pooling
        conv_outs = []
        for conv in self.convs:
            c = F.relu(conv(embedded))        # (batch, num_filters, seq_len - ks + 1)
            c = F.max_pool1d(c, c.size(2))     # (batch, num_filters, 1)
            c = c.squeeze(2)                    # (batch, num_filters)
            conv_outs.append(c)

        # Concatenate all filter outputs
        cat = torch.cat(conv_outs, dim=1)       # (batch, total_filters)
        cat = self.dropout(cat)
        out = self.fc(cat)                       # (batch, output_dim)

        return out


class BugReportEncoder(nn.Module):
    """
    Full bug report encoder: decompose → tokenize → TextCNN per clue.
    Returns list of clue embeddings (not averaged, following paper).

    Args:
        vocab: Vocabulary object
        embed_dim: embedding dimension
        max_clue_len: max token length per clue
        max_clues: max number of clues to keep
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 200,
        num_filters: int = 100,
        kernel_sizes: list[int] = None,
        max_clue_len: int = 256,
        max_clues: int = 5,
    ):
        super().__init__()

        self.max_clue_len = max_clue_len
        self.max_clues = max_clues
        self.embed_dim = embed_dim

        # Shared TextCNN for all clues (weight sharing as in paper)
        self.text_cnn = TextCNN(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_filters=num_filters,
            kernel_sizes=kernel_sizes,
            output_dim=embed_dim,
        )

    def forward(self, clue_indices: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            clue_indices: (batch, max_clues, max_clue_len) padded token indices

        Returns:
            List of clue embeddings, each (batch, embed_dim).
            Number of clues = max_clues (padded clues will have zero embeddings)
        """
        batch_size, num_clues, seq_len = clue_indices.shape
        clue_embs = []

        for i in range(num_clues):
            clue_input = clue_indices[:, i, :]  # (batch, seq_len)
            emb = self.text_cnn(clue_input)      # (batch, embed_dim)
            clue_embs.append(emb)

        return clue_embs
