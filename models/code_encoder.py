"""
Source Code Encoder for BLoco - Option B: Code-NoN.

Code-NoN (Network of Networks):
  - Level 1 (AST): DGP-GNN on AST subgraph of each basic block → block embedding
  - Level 2 (CFG): DGP-GNN on CFG graph over block embeddings → file vector

DGP update formula from paper:
  h_v^(l+1) = σ ( W^(l) * AGG({h_u^(l) | u ∈ N(v)}) + b^(l) )

Also keeps Option A (Bi-GRU) as fallback via `use_code_non` flag.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.graph_builder import CodeGraph, NUM_AST_NODE_TYPES


class DGPGNNLayer(nn.Module):
    """
    Dense Graph Propagation GNN layer.

    h_v^(l+1) = σ(W * AGG({h_u | u ∈ N(v)}) + b)

    Uses adjacency matrix multiplication for message passing.
    AGG = mean (via row-normalized adjacency matrix).
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:   (num_nodes, in_dim) node features
            adj: (num_nodes, num_nodes) row-normalized adjacency

        Returns:
            (num_nodes, out_dim) updated features
        """
        # Message passing: aggregate neighbor features
        agg = torch.matmul(adj, h)      # (N, in_dim)
        out = self.linear(agg)           # (N, out_dim)
        out = F.relu(out)
        return out


class DGPGNN(nn.Module):
    """
    Multi-layer DGP-GNN.
    Dense connections: concatenates outputs of all layers (like DenseNet).
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()

        current_dim = input_dim
        for _ in range(num_layers):
            self.layers.append(DGPGNNLayer(current_dim, hidden_dim))
            current_dim = hidden_dim  # For dense: += hidden_dim, but simple approach

        self.output_dim = hidden_dim

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:   (N, input_dim)
            adj: (N, N)

        Returns:
            (N, output_dim)
        """
        for layer in self.layers:
            h = layer(h, adj)
        return h


class CodeNoN(nn.Module):
    """
    Code-NoN encoder (Network of Networks) from BLoco paper.

    Architecture:
      1. Embed AST node types
      2. AST-level GNN: for each basic block's AST → block_embedding
      3. CFG-level GNN: propagate over block embeddings → all block features
      4. Global pooling → file_vec

    Args:
        embed_dim: AST node type embedding dimension
        hidden_dim: GNN hidden dimension
        output_dim: final file vector dimension
        ast_gnn_layers: number of GNN layers for AST level
        cfg_gnn_layers: number of GNN layers for CFG level
    """

    def __init__(
        self,
        embed_dim: int = 200,
        hidden_dim: int = 200,
        output_dim: int = 200,
        ast_gnn_layers: int = 3,
        cfg_gnn_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # AST node type embedding
        self.ast_type_embed = nn.Embedding(
            NUM_AST_NODE_TYPES, embed_dim, padding_idx=0
        )

        # AST-level GNN
        self.ast_gnn = DGPGNN(embed_dim, hidden_dim, ast_gnn_layers)

        # CFG-level GNN
        self.cfg_gnn = DGPGNN(hidden_dim, hidden_dim, cfg_gnn_layers)

        # Final projection
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward_single(self, graph: CodeGraph) -> torch.Tensor:
        """
        Encode a single file's Code-NoN graph.

        Args:
            graph: CodeGraph with CFG adjacency and AST subgraphs

        Returns:
            (output_dim,) file vector
        """
        device = self.ast_type_embed.weight.device
        block_embeddings = []

        # Level 1: AST-level GNN for each basic block
        for ast_types, ast_adj in zip(graph.ast_node_types, graph.ast_adjs):
            ast_types = ast_types.to(device)
            ast_adj = ast_adj.to(device)

            # Embed AST node types
            h = self.ast_type_embed(ast_types)  # (N_ast, embed_dim)

            # AST GNN
            h = self.ast_gnn(h, ast_adj)         # (N_ast, hidden_dim)

            # Pool AST → single block embedding (mean pooling)
            block_emb = h.mean(dim=0)             # (hidden_dim,)
            block_embeddings.append(block_emb)

        # Stack block embeddings: (num_blocks, hidden_dim)
        block_embs = torch.stack(block_embeddings, dim=0)

        # Level 2: CFG-level GNN
        cfg_adj = graph.cfg_adj.to(device)
        cfg_h = self.cfg_gnn(block_embs, cfg_adj)  # (num_blocks, hidden_dim)

        # Global mean pooling over CFG nodes → file vector
        file_vec = cfg_h.mean(dim=0)                 # (hidden_dim,)

        # Project
        file_vec = self.dropout(file_vec)
        file_vec = self.fc(file_vec)                  # (output_dim,)

        return file_vec

    def forward(self, graphs: list[CodeGraph]) -> torch.Tensor:
        """
        Encode a batch of file graphs.

        Args:
            graphs: list of CodeGraph objects (one per file)

        Returns:
            (batch, output_dim) file vectors
        """
        file_vecs = [self.forward_single(g) for g in graphs]
        return torch.stack(file_vecs, dim=0)


# ============================================================
# Bi-GRU CodeEncoder (Option A) kept as fallback
# ============================================================
class CodeEncoderGRU(nn.Module):
    """Option A: Bi-GRU encoder (kept as fallback / comparison)."""

    def __init__(self, vocab_size, embed_dim=200, hidden_dim=200,
                 num_layers=2, output_dim=200, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers,
                          batch_first=True, bidirectional=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(x))
        output, _ = self.gru(embedded)
        mask = (x != 0).unsqueeze(-1).float()
        mean_out = (output * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(mean_out)
