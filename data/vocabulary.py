"""
Vocabulary builder and text tokenizer for BLoco.
"""
import re
import os
from collections import Counter


# Java keywords for code-aware tokenization
JAVA_KEYWORDS = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
    "class", "const", "continue", "default", "do", "double", "else", "enum",
    "extends", "final", "finally", "float", "for", "goto", "if", "implements",
    "import", "instanceof", "int", "interface", "long", "native", "new",
    "package", "private", "protected", "public", "return", "short", "static",
    "strictfp", "super", "switch", "synchronized", "this", "throw", "throws",
    "transient", "try", "void", "volatile", "while",
}

# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1


def tokenize_text(text: str) -> list[str]:
    """
    Tokenize natural language text (bug reports).
    Splits on whitespace and punctuation, lowercased.
    """
    text = text.lower()
    # Split camelCase and PascalCase
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Split on non-alphanumeric
    tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+', text)
    return tokens


def tokenize_java_code(code: str) -> list[str]:
    """
    Tokenize Java source code.
    Removes comments, splits camelCase, lowercases.
    """
    # Remove block comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    # Remove line comments
    code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
    # Remove string literals
    code = re.sub(r'"[^"]*"', 'STRING_LIT', code)
    code = re.sub(r"'[^']*'", 'CHAR_LIT', code)

    # Split camelCase
    code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
    code = code.lower()

    tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+', code)
    return tokens


class Vocabulary:
    """
    Word vocabulary with frequency-based filtering.
    """

    def __init__(self, max_size: int = 50000, min_freq: int = 2):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.idx2word = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}
        self.word_freq = Counter()

    def build_from_texts(self, texts: list[str], tokenize_fn=tokenize_text):
        """Build vocabulary from a list of raw texts."""
        for text in texts:
            tokens = tokenize_fn(text)
            self.word_freq.update(tokens)

        # Filter by min_freq and take top max_size
        idx = len(self.word2idx)
        for word, freq in self.word_freq.most_common(self.max_size):
            if freq < self.min_freq:
                break
            if word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

        print(f"[Vocabulary] Built vocab: {len(self.word2idx)} words "
              f"(from {len(self.word_freq)} unique tokens)")

    def encode(self, text: str, max_len: int, tokenize_fn=tokenize_text) -> list[int]:
        """Encode text to indices, padded/truncated to max_len."""
        tokens = tokenize_fn(text)
        indices = [self.word2idx.get(t, UNK_IDX) for t in tokens[:max_len]]

        # Pad
        if len(indices) < max_len:
            indices.extend([PAD_IDX] * (max_len - len(indices)))

        return indices

    def __len__(self):
        return len(self.word2idx)

    def save(self, path: str):
        """Save vocabulary to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for word, idx in sorted(self.word2idx.items(), key=lambda x: x[1]):
                f.write(f"{word}\t{idx}\n")

    def load(self, path: str):
        """Load vocabulary from file."""
        self.word2idx = {}
        self.idx2word = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    word, idx = parts[0], int(parts[1])
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
        print(f"[Vocabulary] Loaded vocab: {len(self.word2idx)} words")
