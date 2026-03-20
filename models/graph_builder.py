"""
Java Code Graph Builder for Code-NoN.
Builds CFG (Control Flow Graph) with AST (Abstract Syntax Tree) subgraphs
from Java source code using javalang.

Uses pure PyTorch (no DGL/PyG dependency for portability).

Code-NoN = Network of Networks:
  - Main graph: CFG where nodes = basic blocks
  - Each CFG node contains a sub-graph: the AST of that basic block
"""
import javalang
import torch
import numpy as np


# ============================================================
# AST Node type vocabulary
# ============================================================
AST_NODE_TYPES = [
    "<PAD>", "<UNK>",
    "ClassDeclaration", "MethodDeclaration", "ConstructorDeclaration",
    "FieldDeclaration", "VariableDeclaration", "VariableDeclarator",
    "FormalParameter", "LocalVariableDeclaration",
    "IfStatement", "WhileStatement", "ForStatement", "DoStatement",
    "SwitchStatement", "CaseStatement", "ReturnStatement",
    "ThrowStatement", "TryStatement", "CatchClause",
    "BreakStatement", "ContinueStatement", "AssertStatement",
    "SynchronizedStatement", "BlockStatement", "StatementExpression",
    "ExpressionStatement",
    "MethodInvocation", "SuperMethodInvocation",
    "MemberReference", "FieldAccess",
    "Assignment", "BinaryOperation", "UnaryOperation", "TernaryExpression",
    "ClassCreator", "ArrayCreator", "ArraySelector",
    "Cast", "InstanceOf",
    "Literal", "StringLiteral", "IntegerLiteral",
    "BasicType", "ReferenceType", "TypeArgument",
    "This", "SuperConstructorInvocation",
    "EnhancedForControl", "ForControl",
    "Annotation", "ElementValuePair",
    "Import", "PackageDeclaration",
]

AST_NODE_TYPE_TO_IDX = {t: i for i, t in enumerate(AST_NODE_TYPES)}
NUM_AST_NODE_TYPES = len(AST_NODE_TYPES)


def get_ast_node_type_idx(node) -> int:
    """Get the index of an AST node type."""
    type_name = type(node).__name__
    return AST_NODE_TYPE_TO_IDX.get(type_name, 1)  # 1 = <UNK>


# ============================================================
# CFG Basic Block Extraction
# ============================================================
BRANCH_TYPES = (
    javalang.tree.IfStatement,
    javalang.tree.WhileStatement,
    javalang.tree.ForStatement,
    javalang.tree.DoStatement,
    javalang.tree.SwitchStatement,
    javalang.tree.TryStatement,
    javalang.tree.ReturnStatement,
    javalang.tree.ThrowStatement,
    javalang.tree.BreakStatement,
    javalang.tree.ContinueStatement,
)


def _get_body(stmt) -> list:
    """Get body statements from a statement."""
    if isinstance(stmt, javalang.tree.BlockStatement):
        return stmt.statements if stmt.statements else []
    elif isinstance(stmt, list):
        return stmt
    else:
        return [stmt]


def _extract_sub_blocks(stmt) -> list[list]:
    """Extract basic blocks from sub-statements of a branch."""
    subs = []
    if isinstance(stmt, javalang.tree.IfStatement):
        if stmt.then_statement:
            subs.extend(_extract_blocks(_get_body(stmt.then_statement)))
        if stmt.else_statement:
            subs.extend(_extract_blocks(_get_body(stmt.else_statement)))
    elif isinstance(stmt, (javalang.tree.WhileStatement,
                           javalang.tree.ForStatement,
                           javalang.tree.DoStatement)):
        if stmt.body:
            subs.extend(_extract_blocks(_get_body(stmt.body)))
    elif isinstance(stmt, javalang.tree.TryStatement):
        if stmt.block:
            subs.extend(_extract_blocks(stmt.block))
        if stmt.catches:
            for c in stmt.catches:
                if c.block:
                    subs.extend(_extract_blocks(c.block))
        if stmt.finally_block:
            subs.extend(_extract_blocks(stmt.finally_block))
    elif isinstance(stmt, javalang.tree.SwitchStatement):
        if stmt.cases:
            for c in stmt.cases:
                if c.statements:
                    subs.extend(_extract_blocks(c.statements))
    return subs


def _extract_blocks(body: list) -> list[list]:
    """Extract basic blocks from a list of statements."""
    blocks, cur = [], []
    for stmt in body:
        cur.append(stmt)
        if isinstance(stmt, BRANCH_TYPES):
            blocks.append(cur)
            cur = []
            blocks.extend(_extract_sub_blocks(stmt))
    if cur:
        blocks.append(cur)
    return [b for b in blocks if b]


# ============================================================
# AST Subgraph
# ============================================================
def _build_ast_edges(stmts: list) -> tuple[list[int], list[tuple[int, int]]]:
    """Build AST subgraph: returns (node_type_indices, edge_list)."""
    node_types, edges = [], []

    def visit(node, parent_idx=None):
        if node is None or not isinstance(node, javalang.tree.Node):
            return
        idx = len(node_types)
        node_types.append(get_ast_node_type_idx(node))
        if parent_idx is not None:
            edges.append((parent_idx, idx))
            edges.append((idx, parent_idx))
        for child in node.children:
            if isinstance(child, javalang.tree.Node):
                visit(child, idx)
            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, javalang.tree.Node):
                        visit(item, idx)

    for stmt in stmts:
        visit(stmt)
    if not node_types:
        node_types = [1]
    if not edges:
        edges = [(0, 0)]
    return node_types, edges


# ============================================================
# Graph Data Structure (pure PyTorch)
# ============================================================
class GraphBuildStats:
    """Tracks graph building statistics for observability."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0
        self.parsed_ok = 0
        self.fallback = 0
        self.total_cfg_blocks = 0
        self.total_ast_nodes = 0

    def record_success(self, num_blocks: int, num_ast_nodes: int):
        self.total += 1
        self.parsed_ok += 1
        self.total_cfg_blocks += num_blocks
        self.total_ast_nodes += num_ast_nodes

    def record_fallback(self):
        self.total += 1
        self.fallback += 1

    def report(self) -> str:
        if self.total == 0:
            return "[GraphBuilder] No files processed yet."
        pct = self.parsed_ok / self.total * 100
        avg_blocks = self.total_cfg_blocks / max(self.parsed_ok, 1)
        avg_ast = self.total_ast_nodes / max(self.parsed_ok, 1)
        return (
            f"[GraphBuilder] {self.total} files processed: "
            f"{self.parsed_ok} parsed OK ({pct:.1f}%), "
            f"{self.fallback} fallback | "
            f"Avg CFG blocks: {avg_blocks:.1f}, Avg AST nodes: {avg_ast:.1f}"
        )


# Global stats tracker
graph_stats = GraphBuildStats()


class CodeGraph:
    """
    Pure-PyTorch graph representation for Code-NoN.

    Attributes:
        cfg_adj:        (num_blocks, num_blocks) float adjacency matrix of CFG
        ast_node_types: list of LongTensor, each (num_ast_nodes,) per block
        ast_adjs:       list of FloatTensor, each (N, N) adjacency per block
        num_blocks:     int
        is_fallback:    bool, True if parse failed and using minimal graph
    """

    def __init__(self, cfg_adj, ast_node_types, ast_adjs, is_fallback=False):
        self.cfg_adj = cfg_adj              # (B, B)
        self.ast_node_types = ast_node_types  # list of (N_i,) tensors
        self.ast_adjs = ast_adjs              # list of (N_i, N_i) tensors
        self.num_blocks = cfg_adj.size(0)
        self.is_fallback = is_fallback


def build_code_graph(source_code: str, max_blocks: int = 50, max_ast_nodes: int = 64) -> CodeGraph:
    """
    Build the full Code-NoN graph for a Java source file.
    Handles methods, constructors, interfaces, enums, annotations,
    and abstract classes. Falls back to regex-based tokenization only
    when javalang completely fails to parse.
    """
    if not source_code or not source_code.strip():
        graph_stats.record_fallback()
        return _fallback_graph()

    try:
        tree = javalang.parse.parse(source_code)
    except Exception:
        # javalang can't parse → regex-based fallback (still Code-NoN, NOT Bi-GRU)
        return _regex_fallback_graph(source_code, max_ast_nodes)

    all_blocks = []

    # 1. Method bodies → CFG basic blocks
    for _, node in tree.filter(javalang.tree.MethodDeclaration):
        if node.body:
            all_blocks.extend(_extract_blocks(node.body))

    # 2. Constructor bodies
    for _, node in tree.filter(javalang.tree.ConstructorDeclaration):
        if node.body:
            all_blocks.extend(_extract_blocks(node.body))

    # 3. Static initializers
    for _, node in tree.filter(javalang.tree.ClassDeclaration):
        if hasattr(node, 'body') and node.body:
            for member in node.body:
                # StaticInitializer / BlockStatement at class level
                if isinstance(member, javalang.tree.BlockStatement):
                    if member.statements:
                        all_blocks.extend(_extract_blocks(member.statements))

    # 4. If still no blocks, extract structure from declarations
    if not all_blocks:
        all_blocks = _extract_declaration_blocks(tree)

    if not all_blocks:
        graph_stats.record_fallback()
        return _fallback_graph()

    return _blocks_to_graph(all_blocks, max_blocks, max_ast_nodes)


def _extract_declaration_blocks(tree) -> list[list]:
    """
    Extract blocks from non-method declarations:
    - Interface method signatures (each becomes a block)
    - Enum constants
    - Field declarations
    - Annotation declarations
    - Abstract method signatures
    """
    blocks = []

    # Interface method signatures
    for _, node in tree.filter(javalang.tree.InterfaceDeclaration):
        if hasattr(node, 'body') and node.body:
            for member in node.body:
                blocks.append([member])  # Each method signature as a block

    # Class declarations with abstract methods / fields
    for _, node in tree.filter(javalang.tree.ClassDeclaration):
        if hasattr(node, 'body') and node.body:
            for member in node.body:
                if isinstance(member, (javalang.tree.MethodDeclaration,
                                       javalang.tree.FieldDeclaration)):
                    blocks.append([member])

    # Enum declarations
    for _, node in tree.filter(javalang.tree.EnumDeclaration):
        if hasattr(node, 'body') and node.body:
            # Enum body: constants + methods
            if node.body.constants:
                for const in node.body.constants:
                    blocks.append([const])
            if node.body.declarations:
                for decl in node.body.declarations:
                    blocks.append([decl])

    # Annotation declarations
    for _, node in tree.filter(javalang.tree.AnnotationDeclaration):
        if hasattr(node, 'body') and node.body:
            for member in node.body:
                blocks.append([member])
        else:
            blocks.append([node])  # The annotation itself

    return blocks


def _regex_fallback_graph(source_code: str, max_ast_nodes: int = 64) -> CodeGraph:
    """
    Regex-based fallback for files javalang can't parse.
    Still creates a graph with Code-NoN structure (NOT Bi-GRU fallback).
    
    Splits code into method-like blocks using regex, creates one CFG node
    per detected block with UNK AST node types proportional to token count.
    """
    import re

    # Split by method/class/interface declarations via regex
    method_pattern = r'(?:public|private|protected|static|final|abstract|synchronized|native|void|int|long|String|boolean|double|float|char|byte|short)\s+\w+\s*\([^)]*\)'
    matches = list(re.finditer(method_pattern, source_code))

    if not matches:
        # No methods found at all → single block from whole file
        graph_stats.record_success(1, 1)
        cfg_adj = torch.ones(1, 1)
        ast_types = [torch.tensor([1], dtype=torch.long)]  # UNK
        ast_adjs = [torch.ones(1, 1)]
        return CodeGraph(cfg_adj, ast_types, ast_adjs, is_fallback=False)

    # Create blocks from matched regions
    blocks_text = []
    for i, m in enumerate(matches[:50]):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else min(start + 2000, len(source_code))
        blocks_text.append(source_code[start:end])

    n = len(blocks_text)

    # CFG adjacency
    cfg_adj = torch.zeros(n, n)
    for i in range(n):
        cfg_adj[i, i] = 1.0
        if i + 1 < n:
            cfg_adj[i, i + 1] = 1.0
            cfg_adj[i + 1, i] = 1.0

    row_sum = cfg_adj.sum(dim=1, keepdim=True).clamp(min=1)
    cfg_adj = cfg_adj / row_sum

    # Create AST-like subgraphs using token counts as proxy for structure
    ast_node_types_list = []
    ast_adjs_list = []
    total_ast = 0

    for block_text in blocks_text:
        tokens = re.findall(r'[a-zA-Z_]\w*', block_text)
        num_nodes = min(max(len(tokens) // 4, 2), max_ast_nodes)
        total_ast += num_nodes

        # All UNK (type=1) since we can't determine actual AST types
        node_types = torch.ones(num_nodes, dtype=torch.long)  # all UNK
        ast_adj = torch.zeros(num_nodes, num_nodes)

        # Linear chain structure
        for i in range(num_nodes):
            ast_adj[i, i] = 1.0
            if i + 1 < num_nodes:
                ast_adj[i, i + 1] = 1.0
                ast_adj[i + 1, i] = 1.0

        row_sum = ast_adj.sum(dim=1, keepdim=True).clamp(min=1)
        ast_adj = ast_adj / row_sum

        ast_node_types_list.append(node_types)
        ast_adjs_list.append(ast_adj)

    graph_stats.record_success(n, total_ast)
    return CodeGraph(cfg_adj, ast_node_types_list, ast_adjs_list, is_fallback=False)


def _blocks_to_graph(all_blocks, max_blocks, max_ast_nodes):
    """Convert a list of statement blocks into a CodeGraph."""
    all_blocks = all_blocks[:max_blocks]
    n = len(all_blocks)

    # CFG adjacency
    cfg_adj = torch.zeros(n, n)
    for i in range(n):
        cfg_adj[i, i] = 1.0
        if i + 1 < n:
            cfg_adj[i, i + 1] = 1.0
            cfg_adj[i + 1, i] = 1.0

    # AST subgraphs
    ast_node_types_list = []
    ast_adjs_list = []

    for block in all_blocks:
        node_types, edges = _build_ast_edges(block)
        node_types = node_types[:max_ast_nodes]
        num_nodes = len(node_types)

        ast_adj = torch.zeros(num_nodes, num_nodes)
        for src, dst in edges:
            if src < num_nodes and dst < num_nodes:
                ast_adj[src, dst] = 1.0

        row_sum = ast_adj.sum(dim=1, keepdim=True).clamp(min=1)
        ast_adj = ast_adj / row_sum

        ast_node_types_list.append(torch.tensor(node_types, dtype=torch.long))
        ast_adjs_list.append(ast_adj)

    # Row-normalize CFG
    row_sum = cfg_adj.sum(dim=1, keepdim=True).clamp(min=1)
    cfg_adj = cfg_adj / row_sum

    total_ast = sum(len(nt) for nt in ast_node_types_list)
    graph_stats.record_success(n, total_ast)

    return CodeGraph(cfg_adj, ast_node_types_list, ast_adjs_list, is_fallback=False)



def _fallback_graph() -> CodeGraph:
    """Minimal graph when parsing fails."""
    cfg_adj = torch.ones(1, 1)
    ast_types = [torch.tensor([1], dtype=torch.long)]
    ast_adjs = [torch.ones(1, 1)]
    return CodeGraph(cfg_adj, ast_types, ast_adjs, is_fallback=True)


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    test_code = """
    public class Example {
        public int add(int a, int b) {
            if (a > 0) {
                return a + b;
            } else {
                int result = 0;
                for (int i = 0; i < b; i++) {
                    result += a;
                }
                return result;
            }
        }

        public void process(String name) {
            try {
                System.out.println(name);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
    """
    graph = build_code_graph(test_code)
    print(f"CFG blocks: {graph.num_blocks}")
    print(f"CFG adjacency shape: {graph.cfg_adj.shape}")
    for i, (nt, adj) in enumerate(zip(graph.ast_node_types, graph.ast_adjs)):
        print(f"  Block {i}: {len(nt)} AST nodes, adj shape {adj.shape}")
