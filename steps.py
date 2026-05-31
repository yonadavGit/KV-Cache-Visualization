STEPS = [
    {
        "key": "input_x",
        "title": "Step 1 — Input Embeddings (X)",
        "subtitle": "Shape: seq_len × d_model",
        "explanation": """
Each row is one token embedding.
""",
        "active_nodes": ["X"],
        "active_edges": [],
    },
    {
        "key": "compute_q",
        "title": "Step 2 — Query Matrix (Q = X · Wq)",
        "subtitle": "Shape: seq_len × d_head",
        "explanation": """
Q = X · Wq.
Queries represent what each token is looking for.
""",
        "active_nodes": ["X", "Wq", "Q"],
        "active_edges": ["X->Q", "Wq->Q"],
    },
    {
        "key": "compute_k",
        "title": "Step 3 — Key Matrix (K = X · Wk)",
        "subtitle": "Shape: seq_len × d_head",
        "explanation": """
K = X · Wk.
Keys represent what each token offers for matching.
""",
        "active_nodes": ["X", "Wk", "K"],
        "active_edges": ["X->K", "Wk->K"],
    },
    {
        "key": "compute_v",
        "title": "Step 4 — Value Matrix (V = X · Wv)",
        "subtitle": "Shape: seq_len × d_head",
        "explanation": """
V = X · Wv.
Values are the content that will be mixed into the output.
""",
        "active_nodes": ["X", "Wv", "V"],
        "active_edges": ["X->V", "Wv->V"],
    },
    {
        "key": "show_qkv",
        "title": "Step 5 — Q, K, V Ready",
        "subtitle": "Three views of the same input X",
        "explanation": """
Q and K decide attention.
V carries the content.
""",
        "active_nodes": ["X", "Wq", "Wk", "Wv", "Q", "K", "V"],
        "active_edges": ["X->Q", "Wq->Q", "X->K", "Wk->K", "X->V", "Wv->V"],
    },
    {
        "key": "scores",
        "title": "Step 6 — Raw Attention Scores (S = Q · Kᵀ)",
        "subtitle": "Shape: seq_len × seq_len",
        "explanation": """
S = Q · Kᵀ.
Each cell scores how much one token matches another.
""",
        "active_nodes": ["Q", "K", "S"],
        "active_edges": ["Q->S", "K->S"],
    },
    {
        "key": "scale",
        "title": "Step 7 — Scaled Scores (S / √d_head)",
        "subtitle": "Shape: seq_len × seq_len",
        "explanation": """
Scale scores by √d_head.
Here: √{d_head} ≈ {divisor}.
""",
        "active_nodes": ["S", "S_scaled"],
        "active_edges": ["S->S_scaled"],
    },
    {
        "key": "mask",
        "title": "Step 8 — Causal Mask (S_masked)",
        "subtitle": "Upper triangle → −∞",
        "explanation": """
Future positions are masked out.
Each token can only attend to itself and earlier tokens.
""",
        "active_nodes": ["S_scaled", "S_masked"],
        "active_edges": ["S_scaled->S_masked"],
    },
    {
        "key": "softmax",
        "title": "Step 9 — Attention Weights (A = softmax(S_masked))",
        "subtitle": "Shape: seq_len × seq_len  |  each row sums to 1",
        "explanation": """
Softmax turns each row into attention weights.
Each row sums to 1.
""",
        "active_nodes": ["S_masked", "A"],
        "active_edges": ["S_masked->A"],
    },
    {
        "key": "context",
        "title": "Step 10 — Context Matrix (C = A · V)",
        "subtitle": "Shape: seq_len × d_head",
        "explanation": """
C = A · V.
Each output row is a weighted mix of value vectors.
""",
        "active_nodes": ["A", "V", "C"],
        "active_edges": ["A->C", "V->C"],
    },
    {
        "key": "summary",
        "title": "Step 11 — Full Pipeline",
        "subtitle": "X → Q, K, V → S → scale → mask → A → C",
        "explanation": """
```
Q,K,V = X@Wq, X@Wk, X@Wv
S     = Q@Kᵀ
A     = softmax(mask(S / √d_head))
C     = A@V
```
""",
        "active_nodes": ["X", "Wq", "Wk", "Wv", "Q", "K", "V",
                         "S", "S_scaled", "S_masked", "A", "C"],
        "active_edges": [
            "X->Q", "Wq->Q", "X->K", "Wk->K", "X->V", "Wv->V",
            "Q->S", "K->S",
            "S->S_scaled", "S_scaled->S_masked", "S_masked->A",
            "A->C", "V->C",
        ],
    },
]

STEP_KEYS = [s["key"] for s in STEPS]


def get_step(index: int) -> dict:
    return STEPS[index]


def num_steps() -> int:
    return len(STEPS)
