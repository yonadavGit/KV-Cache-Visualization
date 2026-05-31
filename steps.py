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

POST_CONTEXT_STEPS = [
    {
        "key": "context_input",
        "title": "Post-Context 1 — Context Matrix (C)",
        "subtitle": "Attention output entering the rest of the block",
        "explanation": """
C is the output of attention.
Now it moves through the rest of the Transformer.
""",
        "active_nodes": ["C"],
        "active_edges": [],
    },
    {
        "key": "ffn",
        "title": "Post-Context 2 — Feed-Forward Network",
        "subtitle": "Token-wise nonlinear processing",
        "explanation": """
The FFN is an MLP: linear → activation → linear.
It processes each token row independently.
""",
        "active_nodes": ["C", "MLP", "FFNOut"],
        "active_edges": ["C->MLP", "MLP->FFNOut"],
    },
    {
        "key": "block_out",
        "title": "Post-Context 3 — Residual + LayerNorm",
        "subtitle": "Finish the current Transformer block",
        "explanation": """
The block combines attention output and FFN output.
Residual connections and normalization stabilize the result.
""",
        "active_nodes": ["C", "FFNOut", "BlockOut"],
        "active_edges": ["FFNOut->BlockOut"],
    },
    {
        "key": "stack",
        "title": "Post-Context 4 — More Transformer Blocks",
        "subtitle": "Repeat the block stack",
        "explanation": """
The matrix flows through the remaining Transformer blocks.
Each block keeps contextualizing the token rows.
""",
        "active_nodes": ["BlockOut", "StackOut"],
        "active_edges": ["BlockOut->StackOut"],
    },
    {
        "key": "final_norm",
        "title": "Post-Context 5 — Final LayerNorm",
        "subtitle": "Normalize before prediction",
        "explanation": """
The final hidden states are normalized.
This prepares them for the output projection.
""",
        "active_nodes": ["StackOut", "FinalNorm"],
        "active_edges": ["StackOut->FinalNorm"],
    },
    {
        "key": "logits",
        "title": "Post-Context 6 — Vocabulary Logits",
        "subtitle": "Project hidden states into vocabulary space",
        "explanation": """
Each row becomes scores over possible next tokens.
These raw scores are logits.
""",
        "active_nodes": ["FinalNorm", "Logits"],
        "active_edges": ["FinalNorm->Logits"],
    },
    {
        "key": "last_row",
        "title": "Post-Context 7 — Keep Last Row",
        "subtitle": "Only the final token predicts what comes next",
        "explanation": """
For generation, earlier rows are discarded.
Only the last row predicts the next token.
""",
        "active_nodes": ["Logits", "LastLogits"],
        "active_edges": ["Logits->LastLogits"],
    },
    {
        "key": "probs",
        "title": "Post-Context 8 — Probabilities",
        "subtitle": "Softmax over the last logits vector",
        "explanation": """
Softmax turns the last logits row into probabilities.
The next token is selected from this distribution.
""",
        "active_nodes": ["LastLogits", "Probs"],
        "active_edges": ["LastLogits->Probs"],
    },
    {
        "key": "post_summary",
        "title": "Post-Context 9 — Prediction Path",
        "subtitle": "C → FFN/MLP → blocks → norm → logits → last row → probabilities",
        "explanation": """
```
C -> FFN/MLP -> block stack -> final norm
  -> logits -> last row -> softmax
```
""",
        "active_nodes": ["C", "MLP", "FFNOut", "BlockOut", "StackOut", "FinalNorm",
                         "Logits", "LastLogits", "Probs"],
        "active_edges": ["C->MLP", "MLP->FFNOut", "FFNOut->BlockOut", "BlockOut->StackOut",
                         "StackOut->FinalNorm", "FinalNorm->Logits",
                         "Logits->LastLogits", "LastLogits->Probs"],
    },
]

POST_CONTEXT_STEP_KEYS = [s["key"] for s in POST_CONTEXT_STEPS]

MULTI_HEAD_STEPS = [
    {
        "key": "mh_input",
        "title": "Multi-Head 1 — Shared Input",
        "subtitle": "The same X is sent to multiple attention heads",
        "explanation": """
Each head receives the same input matrix.
Each head has its own learned projections.
""",
        "active_nodes": ["X"],
        "active_edges": [],
    },
    {
        "key": "head_projections",
        "title": "Multi-Head 2 — Independent Heads",
        "subtitle": "Two heads compute separate Q/K/V views",
        "explanation": """
Head 1 and Head 2 use different learned weights.
That lets them look for different token relationships.
""",
        "active_nodes": ["X", "Wq1", "Wk1", "Wv1", "Q1", "K1", "V1",
                         "Wq2", "Wk2", "Wv2", "Q2", "K2", "V2"],
        "active_edges": ["X->Wq1", "Wq1->Q1", "X->Wk1", "Wk1->K1", "X->Wv1", "Wv1->V1",
                         "X->Wq2", "Wq2->Q2", "X->Wk2", "Wk2->K2", "X->Wv2", "Wv2->V2"],
    },
    {
        "key": "head_attention",
        "title": "Multi-Head 3 — Per-Head Attention",
        "subtitle": "Each head builds its own attention pattern",
        "explanation": """
Each head computes attention independently.
The attention weights can differ across heads.
""",
        "active_nodes": ["Q1", "K1", "A1", "Q2", "K2", "A2"],
        "active_edges": ["Q1->A1", "K1->A1", "Q2->A2", "K2->A2"],
    },
    {
        "key": "head_contexts",
        "title": "Multi-Head 4 — Context Per Head",
        "subtitle": "Each head mixes its own values",
        "explanation": """
Each attention matrix multiplies its own V matrix.
The result is one context matrix per head.
""",
        "active_nodes": ["A1", "V1", "C1", "A2", "V2", "C2"],
        "active_edges": ["A1->C1", "V1->C1", "A2->C2", "V2->C2"],
    },
    {
        "key": "concat",
        "title": "Multi-Head 5 — Concatenate Heads",
        "subtitle": "Head contexts are joined column-wise",
        "explanation": """
C1 and C2 are concatenated along the feature dimension.
This combines both heads' views.
""",
        "active_nodes": ["C1", "C2", "Concat"],
        "active_edges": ["C1->Concat", "C2->Concat"],
    },
    {
        "key": "output_projection",
        "title": "Multi-Head 6 — Output Projection",
        "subtitle": "A learned layer mixes the head information",
        "explanation": """
The concatenated heads pass through a learned output projection.
This produces the final multi-head attention output.
""",
        "active_nodes": ["Concat", "Wo", "MHAOut"],
        "active_edges": ["Concat->Wo", "Wo->MHAOut"],
    },
    {
        "key": "mh_summary",
        "title": "Multi-Head 7 — Full Multi-Head Flow",
        "subtitle": "Independent heads → concatenate → output projection",
        "explanation": """
```
Head 1: X -> Q1,K1,V1 -> A1 -> C1
Head 2: X -> Q2,K2,V2 -> A2 -> C2
Concat = [C1 | C2]
MHAOut = Concat @ Wo
```
""",
        "active_nodes": ["X", "Wq1", "Wk1", "Wv1", "Q1", "K1", "V1", "A1", "C1",
                         "Wq2", "Wk2", "Wv2", "Q2", "K2", "V2", "A2", "C2",
                         "Concat", "Wo", "MHAOut"],
        "active_edges": ["X->Wq1", "Wq1->Q1", "X->Wk1", "Wk1->K1", "X->Wv1", "Wv1->V1",
                         "X->Wq2", "Wq2->Q2", "X->Wk2", "Wk2->K2", "X->Wv2", "Wv2->V2",
                         "Q1->A1", "K1->A1", "Q2->A2", "K2->A2",
                         "A1->C1", "V1->C1", "A2->C2", "V2->C2",
                         "C1->Concat", "C2->Concat", "Concat->Wo", "Wo->MHAOut"],
    },
]

MULTI_HEAD_STEP_KEYS = [s["key"] for s in MULTI_HEAD_STEPS]

KV_CACHE_STEPS = [
    {
        "key": "kv_new_token",
        "title": "KV Cache 1 — Same Attention Flow, One New Row",
        "subtitle": "Orange marks the newly appended token row",
        "explanation": """
The matrices are the normal attention matrices.
Orange marks values caused by the newest token.
""",
        "active_nodes": ["X"],
        "active_edges": [],
    },
    {
        "key": "kv_project_new",
        "title": "KV Cache 2 — New Q/K/V Rows",
        "subtitle": "Only the newest token needs fresh projections",
        "explanation": """
The new X row creates new Q, K, and V rows.
Old K/V rows are reused from cache.
""",
        "active_nodes": ["X", "Wq", "Wk", "Wv", "Q", "K", "V"],
        "active_edges": ["X->Q", "Wq->Q", "X->K", "Wk->K", "X->V", "Wv->V"],
    },
    {
        "key": "kv_retrieve_cache",
        "title": "KV Cache 3 — Cached K/V Rows",
        "subtitle": "Blue marks values reused from the previous step",
        "explanation": """
Previous K and V rows are pulled from cache.
They are reused inside the same K and V matrices.
""",
        "active_nodes": ["K", "V"],
        "active_edges": [],
    },
    {
        "key": "kv_scores",
        "title": "KV Cache 4 — One New Score Row",
        "subtitle": "The new query attends over cached K plus new K",
        "explanation": """
Only the newest Q row is needed.
It scores against all K rows: cached blue rows plus the new orange row.
""",
        "active_nodes": ["Q", "K", "S"],
        "active_edges": ["Q->S", "K->S"],
    },
    {
        "key": "kv_scale_mask_softmax",
        "title": "KV Cache 5 — One New Attention Row",
        "subtitle": "Scale, mask, and softmax only the new row",
        "explanation": """
The new score row becomes one attention row.
Older attention rows are not needed for next-token prediction.
""",
        "active_nodes": ["S", "S_scaled", "S_masked", "A"],
        "active_edges": ["S->S_scaled", "S_scaled->S_masked", "S_masked->A"],
    },
    {
        "key": "kv_context",
        "title": "KV Cache 6 — New Context Row",
        "subtitle": "The new attention row mixes cached V plus new V",
        "explanation": """
The newest A row multiplies all V rows.
Cached V rows contribute without being recomputed.
""",
        "active_nodes": ["A", "V", "C"],
        "active_edges": ["A->C", "V->C"],
    },
    {
        "key": "kv_update",
        "title": "KV Cache 7 — Store Updated K and V",
        "subtitle": "The full K/V matrices become next step's cache",
        "explanation": """
After this step, K and V include the new orange row.
Those full K/V matrices are stored for the next token.
""",
        "active_nodes": ["K", "V"],
        "active_edges": [],
    },
    {
        "key": "kv_summary",
        "title": "KV Cache 8 — Cached Inference Step",
        "subtitle": "Same matrices, but old K/V rows are reused",
        "explanation": """
```
new X row -> new Q/K/V rows
old K/V rows -> reused from cache
new A row = softmax(q_new @ K_full.T)
new C row = A_new @ V_full
```
""",
        "active_nodes": ["X", "Wq", "Wk", "Wv", "Q", "K", "V",
                         "S", "S_scaled", "S_masked", "A", "C"],
        "active_edges": ["X->Q", "Wq->Q", "X->K", "Wk->K", "X->V", "Wv->V",
                         "Q->S", "K->S", "S->S_scaled",
                         "S_scaled->S_masked", "S_masked->A", "A->C", "V->C"],
    },
]

KV_CACHE_STEP_KEYS = [s["key"] for s in KV_CACHE_STEPS]


def _steps_for(flow_mode: str):
    if flow_mode == "post_context":
        return POST_CONTEXT_STEPS
    if flow_mode == "multi_head":
        return MULTI_HEAD_STEPS
    if flow_mode == "kv_cache":
        return KV_CACHE_STEPS
    return STEPS


def get_step(index: int, flow_mode: str = "attention") -> dict:
    return _steps_for(flow_mode)[index]


def num_steps(flow_mode: str = "attention") -> int:
    return len(_steps_for(flow_mode))
