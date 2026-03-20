"""
BLoco - Full Model (Option B: Code-NoN)
Combines Bug Report Encoder + Code-NoN + Bi-Affine Scorer.
"""
import torch
import torch.nn as nn

from models.bug_report import BugReportEncoder
from models.code_encoder import CodeNoN, CodeEncoderGRU
from models.biaffine import BiAffineScorer
from models.graph_builder import CodeGraph


class BLocoModel(nn.Module):
    """
    BLoco: Bug Localization with Bug Report Decomposition
    and Code Hierarchical Network.

    Pipeline:
        Bug Report → Decomposition → TextCNN per clue → clue_embs
        Source Code → Code-NoN (CFG + AST + DGP-GNN) → file_vec
        clue_embs + file_vec → Bi-Affine + FFN → Score

    Args:
        bug_vocab_size: vocabulary size for bug report text
        embed_dim: embedding dimension
        num_filters: TextCNN filter count per kernel
        kernel_sizes: TextCNN kernel sizes
        gnn_hidden_dim: GNN hidden dimension
        ast_gnn_layers: AST-level GNN layers
        cfg_gnn_layers: CFG-level GNN layers
        ffn_hidden_dim: FFN hidden dimension
        max_clues: maximum number of clues per bug report
        use_code_non: if True use Code-NoN (Option B), else Bi-GRU (Option A)
        code_vocab_size: needed only when use_code_non=False (Option A)
    """

    def __init__(
        self,
        bug_vocab_size: int,
        embed_dim: int = 200,
        num_filters: int = 100,
        kernel_sizes: list[int] = None,
        gnn_hidden_dim: int = 200,
        ast_gnn_layers: int = 3,
        cfg_gnn_layers: int = 3,
        ffn_hidden_dim: int = 256,
        max_clues: int = 5,
        use_code_non: bool = True,
        code_vocab_size: int = None,
        code_gru_layers: int = 2,
    ):
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [2, 3, 4, 5]

        self.max_clues = max_clues
        self.embed_dim = embed_dim
        self.use_code_non = use_code_non

        # Bug Report Encoder
        self.bug_encoder = BugReportEncoder(
            vocab_size=bug_vocab_size,
            embed_dim=embed_dim,
            num_filters=num_filters,
            kernel_sizes=kernel_sizes,
            max_clues=max_clues,
        )

        # Code Encoder
        if use_code_non:
            self.code_encoder = CodeNoN(
                embed_dim=embed_dim,
                hidden_dim=gnn_hidden_dim,
                output_dim=embed_dim,
                ast_gnn_layers=ast_gnn_layers,
                cfg_gnn_layers=cfg_gnn_layers,
            )
        else:
            assert code_vocab_size is not None, "code_vocab_size required for Option A"
            self.code_encoder = CodeEncoderGRU(
                vocab_size=code_vocab_size,
                embed_dim=embed_dim,
                hidden_dim=gnn_hidden_dim,
                num_layers=code_gru_layers,
                output_dim=embed_dim,
            )

        # Bi-Affine Scorer
        self.scorer = BiAffineScorer(
            embed_dim=embed_dim,
            ffn_hidden_dim=ffn_hidden_dim,
        )

    def forward(
        self,
        clue_indices: torch.Tensor,
        code_input,
    ) -> torch.Tensor:
        """
        Args:
            clue_indices: (batch, max_clues, max_clue_len) bug report clue tokens
            code_input: if use_code_non: list of CodeGraph objects
                        if not: (batch, max_code_len) token indices

        Returns:
            scores: (batch,) relevance scores
        """
        # Encode bug report clues
        clue_embs = self.bug_encoder(clue_indices)   # list of (batch, embed_dim)

        # Encode source code
        if self.use_code_non:
            file_vecs = self.code_encoder(code_input)  # (batch, embed_dim)
        else:
            file_vecs = self.code_encoder(code_input)   # (batch, embed_dim)

        # Compute bi-affine score
        scores = self.scorer(clue_embs, file_vecs)     # (batch,)

        return scores
