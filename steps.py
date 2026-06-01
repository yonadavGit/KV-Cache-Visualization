INTRO_STEPS = [
    {
        "key": "kv_cache_cartoon",
        "title": "KV Caching — Before and After",
        "subtitle": "The intuition: reuse past keys and values instead of recomputing them",
        "explanation": """
KV caching makes decoding faster by storing the K/V rows from previous tokens.
When the next token arrives, the model computes only the new Q/K/V row and reuses the cached K/V history.
""",
        "active_nodes": [],
        "active_edges": [],
    },
]


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
        "active_nodes": ["A", "V", "C", "Wo"],
        "active_edges": ["A->C", "V->C", "C->Wo"],
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

MHA_KV_CACHE_STEPS = [
    {
        "key": "mha_kv_new",
        "title": "MHA KV Cache 1 — Per-Head K/V Caches",
        "subtitle": "Each attention head owns its own K and V cache",
        "explanation": """
In standard multi-head attention, heads do not share K/V.
Each head caches its own K and V rows.
Q rows are not cached; only the newest Q row is used.
Only the newest context row is needed for prediction.
""",
        "active_nodes": ["K1", "V1", "K2", "V2"],
        "active_edges": [],
    },
    {
        "key": "mha_kv_project",
        "title": "MHA KV Cache 2 — New Rows Per Head",
        "subtitle": "The new token adds K/V rows to every head",
        "explanation": """
The new token creates fresh Q/K/V rows for Head 1 and Head 2.
Old K/V rows are reused separately per head.
Old Q rows are not reused.
""",
        "active_nodes": ["X", "Wq1", "Wk1", "Wv1", "Q1", "K1", "V1",
                         "Wq2", "Wk2", "Wv2", "Q2", "K2", "V2"],
        "active_edges": ["X->Wq1", "Wq1->Q1", "X->Wk1", "Wk1->K1", "X->Wv1", "Wv1->V1",
                         "X->Wq2", "Wq2->Q2", "X->Wk2", "Wk2->K2", "X->Wv2", "Wv2->V2"],
    },
    {
        "key": "mha_kv_reuse",
        "title": "MHA KV Cache 3 — Reuse Separate K/V Caches",
        "subtitle": "Blue rows exist once per head",
        "explanation": """
Head 1 reuses K1/V1 cache.
Head 2 reuses K2/V2 cache.
Q1/Q2 are computed fresh only for the newest token.
""",
        "active_nodes": ["K1", "V1", "K2", "V2"],
        "active_edges": [],
    },
    {
        "key": "mha_kv_attend",
        "title": "MHA KV Cache 4 — Per-Head New Attention Rows",
        "subtitle": "Each head scores against its own cached K",
        "explanation": """
Q1 scores against K1.
Q2 scores against K2.
Only the newest Q rows are used for these scores.
""",
        "active_nodes": ["Q1", "K1", "A1", "Q2", "K2", "A2"],
        "active_edges": ["Q1->A1", "K1->A1", "Q2->A2", "K2->A2"],
    },
    {
        "key": "mha_kv_context",
        "title": "MHA KV Cache 5 — Per-Head New Context Rows",
        "subtitle": "Each head mixes its own cached V",
        "explanation": """
A1 mixes V1.
A2 mixes V2.
Only the newest C1/C2 rows continue to the next-token logits.
""",
        "active_nodes": ["A1", "V1", "C1", "A2", "V2", "C2"],
        "active_edges": ["A1->C1", "V1->C1", "A2->C2", "V2->C2"],
    },
    {
        "key": "mha_kv_merge",
        "title": "MHA KV Cache 6 — Merge Head Outputs",
        "subtitle": "Concatenate context rows and project",
        "explanation": """
The head outputs are concatenated as usual.
The cache difference is upstream: MHA stores K/V per head.
""",
        "active_nodes": ["C1", "C2", "Concat", "Wo", "MHAOut"],
        "active_edges": ["C1->Concat", "C2->Concat", "Concat->Wo", "Wo->MHAOut"],
    },
]

MHA_KV_CACHE_STEP_KEYS = [s["key"] for s in MHA_KV_CACHE_STEPS]

MQA_STEPS = [
    {
        "key": "mqa_input",
        "title": "MQA 1 — Shared Input",
        "subtitle": "Same input, multiple query heads",
        "explanation": """
MQA keeps multiple query heads.
The key and value path is shared.
""",
        "active_nodes": ["X"],
        "active_edges": [],
    },
    {
        "key": "mqa_queries",
        "title": "MQA 2 — Separate Query Heads",
        "subtitle": "Each head still asks a different question",
        "explanation": """
Each query head has its own learned Wq.
This preserves different questions per head.
""",
        "active_nodes": ["X", "Wq1", "Wq2", "Q1", "Q2"],
        "active_edges": ["X->Wq1", "Wq1->Q1", "X->Wq2", "Wq2->Q2"],
    },
    {
        "key": "mqa_shared_kv",
        "title": "MQA 3 — Shared K and V",
        "subtitle": "One K/V pair is shared by all query heads",
        "explanation": """
MQA computes one K matrix and one V matrix.
Every query head attends to this same K/V pair.
""",
        "active_nodes": ["X", "Wk", "Wv", "K", "V"],
        "active_edges": ["X->Wk", "Wk->K", "X->Wv", "Wv->V"],
    },
    {
        "key": "mqa_attention",
        "title": "MQA 4 — Heads Attend to Shared K",
        "subtitle": "Q1 and Q2 both score against the same K",
        "explanation": """
The query heads differ.
The keys they look at are shared.
""",
        "active_nodes": ["Q1", "Q2", "K", "A1", "A2"],
        "active_edges": ["Q1->A1", "K->A1", "Q2->A2", "K->A2"],
    },
    {
        "key": "mqa_contexts",
        "title": "MQA 5 — Contexts Use Shared V",
        "subtitle": "Each head mixes the same V matrix",
        "explanation": """
Each attention row pattern can differ.
But both heads mix the same shared V matrix.
""",
        "active_nodes": ["A1", "A2", "V", "C1", "C2"],
        "active_edges": ["A1->C1", "V->C1", "A2->C2", "V->C2"],
    },
    {
        "key": "mqa_merge",
        "title": "MQA 6 — Concatenate and Project",
        "subtitle": "Head contexts merge like normal multi-head attention",
        "explanation": """
C1 and C2 are concatenated.
A learned Wo projection mixes the heads.
""",
        "active_nodes": ["C1", "C2", "Concat", "Wo", "MQAOut"],
        "active_edges": ["C1->Concat", "C2->Concat", "Concat->Wo", "Wo->MQAOut"],
    },
    {
        "key": "mqa_summary",
        "title": "MQA 7 — Full Multi-Query Flow",
        "subtitle": "Many Q heads, one shared K/V cache",
        "explanation": """
```
Q1 = X @ Wq1
Q2 = X @ Wq2
K,V = X @ Wk, X @ Wv   # shared
C1,C2 attend to same K/V
MQAOut = [C1 | C2] @ Wo
```
""",
        "active_nodes": ["X", "Wq1", "Wq2", "Q1", "Q2", "Wk", "Wv", "K", "V",
                         "A1", "A2", "C1", "C2", "Concat", "Wo", "MQAOut"],
        "active_edges": ["X->Wq1", "Wq1->Q1", "X->Wq2", "Wq2->Q2",
                         "X->Wk", "Wk->K", "X->Wv", "Wv->V",
                         "Q1->A1", "K->A1", "Q2->A2", "K->A2",
                         "A1->C1", "V->C1", "A2->C2", "V->C2",
                         "C1->Concat", "C2->Concat", "Concat->Wo", "Wo->MQAOut"],
    },
]

MQA_KV_CACHE_STEPS = [
    {
        "key": "mqa_kv_new",
        "title": "MQA KV Cache 1 — One Shared K/V Cache",
        "subtitle": "Only one K cache and one V cache are stored",
        "explanation": """
MQA shares K and V across query heads.
So the cache stores one K/V pair, not one pair per head.
Q rows are not cached; only the newest Q rows are used.
Only the newest context rows are needed for prediction.
""",
        "active_nodes": ["X", "K", "V"],
        "active_edges": [],
    },
    {
        "key": "mqa_kv_project",
        "title": "MQA KV Cache 2 — New Rows",
        "subtitle": "New token creates Q rows plus one shared K/V row",
        "explanation": """
The new token creates Q1 and Q2 rows.
It also creates one shared K row and one shared V row.
The Q rows are temporary for this decode step.
""",
        "active_nodes": ["X", "Wq1", "Wq2", "Q1", "Q2", "Wk", "Wv", "K", "V"],
        "active_edges": ["X->Wq1", "Wq1->Q1", "X->Wq2", "Wq2->Q2",
                         "X->Wk", "Wk->K", "X->Wv", "Wv->V"],
    },
    {
        "key": "mqa_kv_reuse",
        "title": "MQA KV Cache 3 — Reused Shared K/V Rows",
        "subtitle": "Blue rows are reused by both query heads",
        "explanation": """
Previous K and V rows are reused once.
Both query heads attend to that same cached K/V.
Previous Q rows are not stored or reused.
""",
        "active_nodes": ["K", "V", "Q1", "Q2"],
        "active_edges": [],
    },
    {
        "key": "mqa_kv_attend",
        "title": "MQA KV Cache 4 — New Attention Rows",
        "subtitle": "Each query head computes one new attention row",
        "explanation": """
Q1 and Q2 each score against the same full K matrix.
Only the newest attention rows are needed.
Those scores use only the newest Q rows.
""",
        "active_nodes": ["Q1", "Q2", "K", "A1", "A2"],
        "active_edges": ["Q1->A1", "K->A1", "Q2->A2", "K->A2"],
    },
    {
        "key": "mqa_kv_context",
        "title": "MQA KV Cache 5 — New Context Rows",
        "subtitle": "Both heads mix the same cached V",
        "explanation": """
A1 and A2 can differ.
Both multiply the same full V matrix.
Only the newest C1/C2 rows continue to the next-token logits.
""",
        "active_nodes": ["A1", "A2", "V", "C1", "C2"],
        "active_edges": ["A1->C1", "V->C1", "A2->C2", "V->C2"],
    },
    {
        "key": "mqa_kv_merge",
        "title": "MQA KV Cache 6 — Merge Head Outputs",
        "subtitle": "Concatenate new context rows and project",
        "explanation": """
The head contexts are still concatenated.
The output projection works as usual.
""",
        "active_nodes": ["C1", "C2", "Concat", "Wo", "MQAOut"],
        "active_edges": ["C1->Concat", "C2->Concat", "Concat->Wo", "Wo->MQAOut"],
    },
]

MQA_STEP_KEYS = [s["key"] for s in MQA_STEPS]
MQA_KV_CACHE_STEP_KEYS = [s["key"] for s in MQA_KV_CACHE_STEPS]

GQA_KV_CACHE_STEPS = [
    {
        "key": "gqa_kv_groups",
        "title": "GQA KV Cache 1 — Four Query Heads, Two K/V Groups",
        "subtitle": "Heads 1-2 share Group 1 K/V; Heads 3-4 share Group 2 K/V",
        "explanation": """
GQA keeps separate query heads.
K and V are shared within groups, not across all heads.
The cache stores one K/V pair per group.
""",
        "active_nodes": ["X", "KG1", "VG1", "KG2", "VG2"],
        "active_edges": [],
    },
    {
        "key": "gqa_kv_project",
        "title": "GQA KV Cache 2 — New Query Rows and Group K/V Rows",
        "subtitle": "The new token adds four Q rows and two grouped K/V rows",
        "explanation": """
Each query head gets a fresh newest Q row.
Each K/V group gets one fresh newest K row and V row.
Older Q rows are not cached.
""",
        "active_nodes": ["X", "Wq1", "Q1", "Wq2", "Q2", "Wq3", "Q3", "Wq4", "Q4",
                         "WkG1", "WvG1", "KG1", "VG1", "WkG2", "WvG2", "KG2", "VG2"],
        "active_edges": ["X->Wq1", "Wq1->Q1", "X->Wq2", "Wq2->Q2",
                         "X->Wq3", "Wq3->Q3", "X->Wq4", "Wq4->Q4",
                         "X->WkG1", "WkG1->KG1", "X->WvG1", "WvG1->VG1",
                         "X->WkG2", "WkG2->KG2", "X->WvG2", "WvG2->VG2"],
    },
    {
        "key": "gqa_kv_reuse",
        "title": "GQA KV Cache 3 — Reuse Cached K/V by Group",
        "subtitle": "Blue rows are cached once per K/V group",
        "explanation": """
Heads 1 and 2 reuse Group 1 K/V cache.
Heads 3 and 4 reuse Group 2 K/V cache.
The cache is smaller than MHA but larger than MQA.
""",
        "active_nodes": ["KG1", "VG1", "KG2", "VG2", "Q1", "Q2", "Q3", "Q4"],
        "active_edges": [],
    },
    {
        "key": "gqa_kv_attend",
        "title": "GQA KV Cache 4 — Heads Attend to Their Group K",
        "subtitle": "Each newest Q row scores against its group's full K matrix",
        "explanation": """
Q1 and Q2 score against KG1.
Q3 and Q4 score against KG2.
Only newest attention rows are needed for cached inference.
""",
        "active_nodes": ["Q1", "Q2", "Q3", "Q4", "KG1", "KG2", "A1", "A2", "A3", "A4"],
        "active_edges": ["Q1->A1", "KG1->A1", "Q2->A2", "KG1->A2",
                         "Q3->A3", "KG2->A3", "Q4->A4", "KG2->A4"],
    },
    {
        "key": "gqa_kv_context",
        "title": "GQA KV Cache 5 — Context Rows Per Query Head",
        "subtitle": "Each head mixes the V matrix from its group",
        "explanation": """
A1 and A2 mix VG1.
A3 and A4 mix VG2.
Only the newest C rows continue.
""",
        "active_nodes": ["A1", "A2", "A3", "A4", "VG1", "VG2", "C1", "C2", "C3", "C4"],
        "active_edges": ["A1->C1", "VG1->C1", "A2->C2", "VG1->C2",
                         "A3->C3", "VG2->C3", "A4->C4", "VG2->C4"],
    },
    {
        "key": "gqa_kv_merge",
        "title": "GQA KV Cache 6 — Concatenate Four Heads",
        "subtitle": "The four newest context rows become one wider concat row",
        "explanation": """
Concat joins C1, C2, C3, and C4 column-wise.
The newest concat row is projected by the learned output weight.
""",
        "active_nodes": ["C1", "C2", "C3", "C4", "Concat", "Wo", "GQAOut"],
        "active_edges": ["C1->Concat", "C2->Concat", "C3->Concat", "C4->Concat",
                         "Concat->Wo", "Wo->GQAOut"],
    },
]

GQA_KV_CACHE_STEP_KEYS = [s["key"] for s in GQA_KV_CACHE_STEPS]

MLA_KV_CACHE_STEPS = [
    {
        "key": "mla_cache_rule",
        "title": "MLA KV Cache 1 — Cache the Latent Matrix",
        "subtitle": "MLA stores cKV, not K or V",
        "explanation": """
MLA compresses X into cKV.
Only cKV is cached between decode steps.
K and V are reconstructed when needed.
""",
        "active_nodes": ["X", "Wdkv", "cKV"],
        "active_edges": ["X->Wdkv", "Wdkv->cKV"],
    },
    {
        "key": "mla_new_token",
        "title": "MLA KV Cache 2 — New Query and New Latent Row",
        "subtitle": "The new token creates q_new and cKV_new",
        "explanation": """
Q is computed fresh for the newest token.
The newest cKV row is appended to the latent cache.
Q is not cached.
""",
        "active_nodes": ["X", "Wq", "Q", "Wdkv", "cKV"],
        "active_edges": ["X->Wq", "Wq->Q", "X->Wdkv", "Wdkv->cKV"],
    },
    {
        "key": "mla_reconstruct",
        "title": "MLA KV Cache 3 — Conceptual K/V Reconstruction",
        "subtitle": "The plot shows decompression; inference can fold fixed projectors",
        "explanation": """
cKV multiplies learned up-projectors Wuk and Wuv.
This reconstructs K and V on the fly.
K and V are used for attention, but they are not cached.

The next steps show the absorption trick:
fixed learned products can be folded so inference works directly with cKV.
""",
        "active_nodes": ["cKV", "Wuk", "Wuv", "K", "V"],
        "active_edges": ["cKV->Wuk", "Wuk->K", "cKV->Wuv", "Wuv->V"],
    },
    {
        "key": "mla_scores",
        "title": "MLA KV Cache 4 — Scores With Absorbed Wuk",
        "subtitle": "Use X · (Wq · Wuk.T) against cKV.T",
        "explanation": """
Conceptually, the newest Q row scores against reconstructed K.
With absorption, the key up-projector is folded into the query side.

Absorption trick for scores:
`Q @ K.T = X @ Wq @ Wuk.T @ cKV.T`.
Since `Wq` and `Wuk` are fixed learned weights, `Wq @ Wuk.T` can be precomputed and reused during inference.
The token-dependent operands are `X`/`q_new` and cached `cKV`.
""",
        "active_nodes": ["X", "Wq", "Q", "Wuk", "cKV", "K", "S", "S_scaled", "S_masked", "A"],
        "active_edges": ["Q->S", "K->S", "S->S_scaled", "S_scaled->S_masked", "S_masked->A"],
    },
    {
        "key": "mla_context",
        "title": "MLA KV Cache 5 — Output With Absorbed Wuv",
        "subtitle": "Use A · cKV · (Wuv · Wo)",
        "explanation": """
Conceptually, the newest A row multiplies reconstructed V.
With absorption, the value up-projector is folded into the output projection.

Value/output-side absorption:
`A @ V @ Wo = A @ cKV @ (Wuv @ Wo)`.
Since `Wuv` and `Wo` are fixed learned weights, `Wuv @ Wo` can be precomputed and reused during inference.
Only the newest output row continues to next-token prediction.
""",
        "active_nodes": ["A", "cKV", "Wuv", "V", "C", "Wo"],
        "active_edges": ["A->C", "V->C", "C->Wo"],
    },
    {
        "key": "mla_summary",
        "title": "MLA KV Cache 6 — Full Latent Cache Flow",
        "subtitle": "Cache cKV; fold fixed learned products during inference",
        "explanation": """
```
q_new   = x_new @ Wq
ckv_new = x_new @ Wdkv
cKV cache = append(ckv_new)

conceptual:
K,V = cKV @ Wuk, cKV @ Wuv
A_new = softmax(q_new @ K.T)
C_new = A_new @ V

absorbed:
scores = X @ (Wq @ Wuk.T) @ cKV.T
output = A @ cKV @ (Wuv @ Wo)
```

Absorbed fixed products:
`Wq @ Wuk.T` for score calculation.
`Wuv @ Wo` for value/output calculation.
""",
        "active_nodes": ["X", "Wq", "Q", "Wdkv", "cKV", "Wuk", "Wuv",
                         "K", "V", "S", "S_scaled", "S_masked", "A", "C", "Wo"],
        "active_edges": ["X->Wq", "Wq->Q", "X->Wdkv", "Wdkv->cKV",
                         "cKV->Wuk", "Wuk->K", "cKV->Wuv", "Wuv->V",
                         "Q->S", "K->S", "S->S_scaled",
                         "S_scaled->S_masked", "S_masked->A", "A->C", "V->C", "C->Wo"],
    },
]

MLA_KV_CACHE_STEP_KEYS = [s["key"] for s in MLA_KV_CACHE_STEPS]

KV_CACHE_STEPS = [
    {
        "key": "kv_new_token",
        "title": "KV Cache 1 — Same Attention Flow, One New Row",
        "subtitle": "Orange marks the newly appended token row",
        "explanation": """
The matrices are the normal attention matrices.
Orange marks values caused by the newest token.
Only K and V are cached; old Q rows are not stored.
Only the newest C row is needed for prediction.
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
The new Q row is used immediately, then discarded.
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
Q is not part of the cache.
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
Older Q rows are not used in cached decode.
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
Only the newest C row continues to the next-token logits.
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
old Q rows -> not cached
new A row = softmax(q_new @ K_full.T)
new C row = A_new @ V_full
old C rows -> not needed for next-token prediction
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
    if flow_mode == "intro":
        return INTRO_STEPS
    if flow_mode == "post_context":
        return POST_CONTEXT_STEPS
    if flow_mode == "multi_head":
        return MULTI_HEAD_STEPS
    if flow_mode == "mha_kv_cache":
        return MHA_KV_CACHE_STEPS
    if flow_mode == "mqa":
        return MQA_STEPS
    if flow_mode == "mqa_kv_cache":
        return MQA_KV_CACHE_STEPS
    if flow_mode == "gqa_kv_cache":
        return GQA_KV_CACHE_STEPS
    if flow_mode == "mla_kv_cache":
        return MLA_KV_CACHE_STEPS
    if flow_mode == "kv_cache":
        return KV_CACHE_STEPS
    return STEPS


def get_step(index: int, flow_mode: str = "attention") -> dict:
    return _steps_for(flow_mode)[index]


def num_steps(flow_mode: str = "attention") -> int:
    return len(_steps_for(flow_mode))
