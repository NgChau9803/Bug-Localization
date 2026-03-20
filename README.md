# BLoco — Bug Localization with Bug Report Decomposition and Code Hierarchical Network

Reproduction of the **BLoco** model from *Knowledge-Based Systems 2022* (Ziye Zhu et al.) for file-level bug localization.

> **Goal:** Given a bug report, rank all source files by likelihood of being buggy.

## Architecture

```
Bug Report → Decompose into clues → TextCNN per clue → clue_embs
                                                          ↘
                                                      Bi-Affine + FFN → Score → Rank files
                                                          ↗
Java Source File → CFG + AST subgraphs → DGP-GNN (Code-NoN) → file_vec
```

**Two modes:**
| Mode | Code Encoder | Flag |
|------|-------------|------|
| **Code-NoN** (paper) | CFG + AST + DGP-GNN | `--mode code_non` |
| **Bi-GRU** (baseline) | Bidirectional GRU | `--mode gru` |

## Project Structure

```
├── config.py              # Hyperparameters & dataset paths
├── train.py               # Training script
├── evaluate.py            # Evaluation (Top-K, MRR, MAP)
├── run_experiments.py     # Train + eval all projects
├── requirements.txt       # Dependencies
├── data/
│   ├── data_loader.py     # XML bug report parser & file indexer
│   ├── dataset.py         # PyTorch Dataset with negative sampling
│   └── vocabulary.py      # Tokenizer & vocabulary builder
├── models/
│   ├── bug_report.py      # Bug report decomposition + TextCNN
│   ├── graph_builder.py   # CFG + AST graph construction (pure PyTorch)
│   ├── code_encoder.py    # Code-NoN (DGP-GNN) & Bi-GRU fallback
│   ├── biaffine.py        # Bi-affine scorer + FFN fusion
│   └── bloco.py           # Full BLoco model
├── scripts/
│   └── diagnose_parse.py  # Graph builder diagnostic tool
└── Datasets/              # (not in git — see below)
    ├── bug reports/       # XML/TXT/XLSX per project
    └── source files/      # Java source trees per project
```

## Datasets

5 Java projects (not included in repo due to size):

| Project | Bug Reports | Java Files |
|---------|-------------|------------|
| AspectJ | 593 | 6,910 |
| Birt | ~4,000+ | 9,697 |
| Eclipse Platform UI | ~6,000+ | 6,165 |
| SWT | ~3,500+ | 2,176 |
| Tomcat | 1,056 | 1,794 |

**Setup:** Place the `Datasets/` folder in the project root with `bug reports/` and `source files/` subdirectories.

## Quick Start

```bash
# 1. Setup environment
pyenv local 3.11.6          # or any Python 3.10+
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train on a single project
python train.py --project Tomcat --mode code_non --epochs 30

# 4. Evaluate
python evaluate.py --project Tomcat --mode code_non

# 5. Run all experiments
python run_experiments.py --mode code_non --epochs 30
```

## Training Options

```
python train.py --help

--project       Project name: Tomcat, AspectJ, Birt, Eclipse_Platform_UI, SWT
--mode          code_non (paper) or gru (baseline)
--epochs        Number of epochs (default: 30)
--batch_size    Batch size (default: 16)
--lr            Learning rate (default: 0.001)
--neg_ratio     Negative samples per positive (default: 5)
--gnn_layers    GNN layers for Code-NoN (default: 3)
--max_bugs      Limit bug reports for debugging
```

## Metrics

- **Top-K Accuracy** (K=1, 5, 10): Is a buggy file in the top K?
- **MRR**: Mean Reciprocal Rank of first relevant file
- **MAP**: Mean Average Precision

## Reference

```
Ziye Zhu et al. "BLoco: Enhancing Bug Localization with Bug Report
Decomposition and Code Hierarchical Network."
Knowledge-Based Systems, 2022. DOI: 10.1016/j.knosys.2022.108741
```
