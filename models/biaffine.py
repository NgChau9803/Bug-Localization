"""
Bi-affine Scoring and FFN Fusion for BLoco.
Following paper Section 4: Relationship Prediction.
"""
import torch
import torch.nn as nn


class BiAffineScorer(nn.Module):
    """
    Bi-affine attention scorer between bug clue embeddings and file vectors.

    Score formula:
        e_ij = c_i^T * W * p_j + b       (bi-affine score per clue-file pair)
        s_j  = sum_i(e_ij) + FFN(concat(sum_i(c_i), p_j))  (final score per file)

    Args:
        embed_dim: dimension of clue/file embeddings
        ffn_hidden_dim: hidden dimension of FFN
    """

    def __init__(self, embed_dim: int = 200, ffn_hidden_dim: int = 256):
        super().__init__()

        # Bi-affine weight matrix W and bias
        self.W = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))

        # FFN: concat(sum_clues, file_vec) → score
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim * 2, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(ffn_hidden_dim, 1),
        )

    def forward(
        self,
        clue_embs: list[torch.Tensor],
        file_vec: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            clue_embs: list of K tensors, each (batch, embed_dim)
            file_vec: (batch, embed_dim)

        Returns:
            score: (batch,) relevance score for each file
        """
        # Bi-affine scores
        biaffine_scores = []
        for c in clue_embs:
            # c: (batch, embed_dim)
            # c @ W: (batch, embed_dim)
            # (c @ W) * p_j → sum over embed_dim → scalar per sample
            cW = torch.matmul(c, self.W)                  # (batch, embed_dim)
            e = (cW * file_vec).sum(dim=-1) + self.b       # (batch,)
            biaffine_scores.append(e)

        # Sum bi-affine scores across all clues
        biaffine_total = torch.stack(biaffine_scores, dim=0).sum(dim=0)  # (batch,)

        # FFN component: concat sum of clues and file vec
        sum_clues = torch.stack(clue_embs, dim=0).sum(dim=0)  # (batch, embed_dim)
        ffn_input = torch.cat([sum_clues, file_vec], dim=-1)   # (batch, embed_dim*2)
        ffn_score = self.ffn(ffn_input).squeeze(-1)             # (batch,)

        # Final score
        score = biaffine_total + ffn_score  # (batch,)

        return score
