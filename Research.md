# BLoco - Enhancing Bug Localization with Bug Report Decomposition and Code Hierarchical Network
Paper: Knowledge-Based Systems 2022 | Ziye Zhu et al. | DOI: 10.1016/j.knosys.2022.108741
Mục tiêu: File-level bug localization (rank list file buggy từ 1 bug report)
Điểm mạnh: Decomposition bug report + Code-NoN (hierarchical graph) + Bi-affine classifier
PDF: https://pdf.sciencedirectassets.com/271505/1-s2.0-S0950705122X00105/1-s2.0-S0950705122003483/am.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEHAaCXVzLWVhc3QtMSJHMEUCIDOxZotmEj5iYqhxr3s%2BQBh0eO%2B3Vq6wcfpOiAjFtGmoAiEAgm3pN69dhtzfAYY5yOtbeEmMDZSsavnykxNUuBzrPZ4qsgUIORAFGgwwNTkwMDM1NDY4NjUiDBL8ojDNK9Hb%2FFI5aiqPBbjns6ZC4fjTwHNx910qq9oTZlU1LJ0%2BXT7PY7S0fQneLW5yFyuMMiGUfXEkjBnpPjMforqFl7mfU9Q6FsKC9k2WVIM0W9dJ%2FCuSgBPQoRcZFaA8GHRExplolJQepE5ZckS1TiqVtP6AkYugkamF8KndzDlxUrFsmvv5t%2BsjJH1jKECKJVh%2B4o2ZY%2BehP2vBbOitJfaGbATfHphnQWQ2M7MDG0J2P5jcH%2BI3Z%2BaUlm0l1WUanD%2FQQsl5OUIujSSOA4o0wvl4gVZl26NKDhV78PXRoB3XrJEWI8mAczx%2Bs3BdlzIiEogn7wrtdug1mCSuagumNy4184hZ9TxRKnEW7iyfAmJjWWfDZntriyLQu64Ry4ppdrAQPGYEBVv2IG5N9VgOvDLhRRGymdKzSJ9dllMtoya8XN3fQMGl%2FQrpkxXkswQ8dI%2BPSzO99Z04RXVJkYAO%2FNXPP0MnSQ2Oz9q6sFHLTpLLunpgWstZRL72HZkiEt5YracTsLizV5nUfAol3Qq%2B0JwUoZmAil2yvPshkrrWBAi23eZkClm9%2Bihsn5lx3EkXix6oCpatk8glD5WXHts5dzkrebsIGbKeSAo1%2BOBNkl0UZgH2eR3QQTr1cWoyZM3Q810FDQlxe0FPwEX6OdUYJSdOrOOnWV15XcZg71mVFh60kQ9cdJqu5mSR4D461WWRZfHt4LPc6fpyOWVYUNVB%2FFR3i8LdJlZp3YjD57aLeTL0eoQubejCD9MuMjwB%2FSLlpPi9J2bWCg1M62iw2ltkqRJCVwRbqwNTa%2FWBarHKfIjIK2j6AYPKyv1hPb%2BzfcsUglejkiRUk50Nte8Qd3sxyrA%2BurLl%2B1aDaYqI8k7ZbHNHprB3E4nn8U5UVB0w4d71zQY6sQHDCayhAJJmk4d0o1mBvxfJBfxjvWhtsN7NJHE1RnxYKwl6m0ulhNIr9rNi6z4U4UdE63apw%2FxibCbTyvUHbE41nQnhPp3PW8L4522xFjj9juw68xmg2B5GOsucxjyTqhU7eu7w%2FzsNhgmLD9fDMKcyfQPihsw6rf2vwVQdSDgqKiEyHreMdgynSnFOH9WaHrIKYi48v8Vifa%2FfjpMih7PFj1hqbTDt6ZGq9%2BY01miWKeQ%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20260320T161844Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYQNEC4TM7%2F20260320%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=3159739c85b3bbfb6160950e466002ac84f293f9fcec20d19fa62b62714a5118&hash=13b34d5ddb1d51da60c0bde3ea65785094dd50c45dd99a74cce4417edf43b468&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0950705122003483&tid=pdf-76712ec2-5078-4f20-bf81-58fc233fa94d&sid=a599e20398178748728af526e6d5ad617390gxrqa&type=client

## 1. Tổng quan Architecture

3 phần chính:
- Bug Report Processing
- Source Code Processing (Code-NoN)
- Relationship Prediction (Bi-affine)

Pipeline:
Bug Report → Decomposition → Multiple Bug Clues → TextCNN per clue → clue_embs (c1, c2, ..., ck)
Source Code Files → Code-NoN Graph (CFG basic blocks + AST subgraphs) → DGP-GNN → file_vec (p)
clue_embs + file_vec → Bi-affine + FFN → Score sj → Rank top-k buggy files

## 2. Bug Report Decomposition & Processing

Decomposition: Chia bug report thành nhiều bug clues riêng biệt (thường 4-6)
Ví dụ clues: Summary, Description/Observed, Steps to reproduce, Expected, Stack trace, Patch

Cách implement gợi ý:
def decompose_bug_report(text: str) -> list[str]:
    clues = []
    # regex hoặc split keyword
    clues.append(extract_summary_and_desc(text))
    clues.append(extract_steps(text))
    clues.append(extract_stack_trace(text))
    clues.append(extract_expected(text))
    # thêm patch nếu có
    return clues

Encoding: Mỗi clue → TextCNN (Conv1D nhiều kernel size + maxpool)
Quan trọng: giữ clue_embs là list riêng, không average sớm

## 3. Code Hierarchical Network – Code-NoN

Tên: Code-NoN = Network of Networks
Cấu trúc:
- Main graph: CFG, nodes = Basic blocks (đoạn code không branch)
- Mỗi node có sub-graph: AST của chính block đó

Xây graph: Dùng JavaParser / Joern / srcML extract CFG + AST

Learning: Multilayer Dense Graph Propagation (DGP-GNN)

Công thức DGP update:
h_v^(l+1) = σ ( W^(l) * AGG({h_u^(l) | u ∈ N(v)}) + b^(l) )
(AGG = mean hoặc sum, σ = ReLU, layers 3-4)

Pseudocode:
class CodeNoN(nn.Module):
    def forward(self, file_code):
        g = build_cfg_with_ast_subgraphs(file_code)  # DGLGraph
        h = dgp_gnn_layers(g, num_layers=3)
        file_vec = global_mean_pool(h)  # hoặc hierarchical pooling
        return file_vec

## 4. Fusion & Prediction

Bi-affine score giữa clue i và file j:
e_ij = c_i^T * W * p_j + b

Final score cho file j:
s_j = sum_i e_ij + FFN( sum_i c_i , sum_k p_k )

Loss:
L = -sum [ y log(y_hat) + (1-y) log(1-y_hat) ]
(có thể thay ListMLE ranking loss)

Class gợi ý:
class BiAffine(nn.Module):
    def forward(self, clue_embs, file_vec):
        scores = []
        for c in clue_embs:
            e = torch.matmul(c.T, self.W) @ file_vec + self.b
            scores.append(e)
        total_score = torch.sum(torch.stack(scores)) + self.ffn(torch.cat([sum(clue_embs), file_vec]))
        return total_score

## 5. Implement Details & Hyper

Framework: PyTorch + DGL/PyG
Embedding dim: 128-300
GNN layers: 3-4
Optimizer: Adam, LR=0.001
Batch: 32-64
Datasets gợi ý: Eclipse, AspectJ, Tomcat (có oracle file)
Metrics: Top-1/5/10, MRR, MAP

## 6. Full Training Skeleton (copy paste chạy luôn)

for epoch in range(epochs):
    for bug_report, candidate_files, labels in dataloader:
        clue_embs = decompose_and_encode(bug_report)          # list tensor
        file_vecs = [code_non_model(f) for f in candidate_files]
        
        scores = [bi_affine(clue_embs, fv) for fv in file_vecs]
        loss = nn.BCEWithLogitsLoss()(torch.stack(scores), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

Hướng dẫn bắt đầu:
1. Làm decompose + TextCNN trước (torch.nn.Conv1d)
2. Xây CodeNoN với DGL (tìm tutorial "build CFG DGL")
3. Thêm BiAffine class
4. Train trên dataset nhỏ của bro
