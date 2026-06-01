import streamlit as st
from attention_math import (
    make_matrices,
    compute_attention,
    compute_post_context,
    compute_multi_head,
    compute_kv_cache_demo,
    compute_multi_query_attention,
    compute_grouped_query_attention,
    compute_mla_attention,
)
from steps import num_steps


def _recompute():
    ss = st.session_state
    X, Wq, Wk, Wv = make_matrices(ss.seq_len, ss.d_model, ss.d_head, seed=ss.seed)
    Q, K, V, S, S_scaled, S_masked, A, C = compute_attention(X, Wq, Wk, Wv)
    ss.matrices = dict(X=X, Wq=Wq, Wk=Wk, Wv=Wv, Q=Q, K=K, V=V,
                       S=S, S_scaled=S_scaled, S_masked=S_masked, A=A, C=C)
    ss.multi_head_matrices = compute_multi_head(X, seed=ss.seed, d_head=ss.d_head)
    ss.mqa_matrices = compute_multi_query_attention(X, seed=ss.seed, d_head=ss.d_head)
    ss.gqa_matrices = compute_grouped_query_attention(X, seed=ss.seed, d_head=ss.d_head)
    ss.mla_matrices = compute_mla_attention(X, seed=ss.seed, d_head=ss.d_head)
    ss.kv_cache_matrices = compute_kv_cache_demo(X, Wq, Wk, Wv)
    ss.post_matrices = compute_post_context(
        C,
        seed=ss.seed,
        vocab_size=ss.post_vocab_size,
        ffn_hidden_mult=ss.ffn_hidden_mult,
    )


def init_state():
    ss = st.session_state
    if "initialized" not in ss:
        ss.seq_len = 4
        ss.d_model = 8
        ss.d_head = 4
        ss.ffn_hidden_mult = 2
        ss.post_vocab_size = 12
        ss.seed = 42
        ss.step_idx = 0
        ss.flow_mode = "intro"
        ss.mla_show_absorption = False
        ss.decode_row_view = False
        ss.tokens = ["The", "cat", "sat", "mat"]
        ss.initialized = True
        _recompute()
    post_defaults_changed = False
    if "ffn_hidden_mult" not in ss:
        ss.ffn_hidden_mult = 2
        post_defaults_changed = True
    if "post_vocab_size" not in ss:
        ss.post_vocab_size = 12
        post_defaults_changed = True
    if "flow_mode" not in ss:
        ss.flow_mode = "intro"
    if "mla_show_absorption" not in ss:
        ss.mla_show_absorption = False
    if "decode_row_view" not in ss:
        ss.decode_row_view = False
    if (post_defaults_changed or "multi_head_matrices" not in ss
            or "mqa_matrices" not in ss or "gqa_matrices" not in ss
            or "mla_matrices" not in ss or "kv_cache_matrices" not in ss):
        _recompute()


def regenerate():
    ss = st.session_state
    ss.seed = (ss.seed + 1) % 10000
    _recompute()


def apply_dim_change():
    ss = st.session_state
    tokens = ss.tokens
    n = ss.seq_len
    if len(tokens) < n:
        default = ["tok" + str(i) for i in range(len(tokens), n)]
        tokens = tokens + default
    ss.tokens = tokens[:n]
    _recompute()


def apply_post_change():
    _recompute()


def go_next():
    ss = st.session_state
    ss.step_idx = min(ss.step_idx + 1, num_steps(ss.flow_mode) - 1)


def go_prev():
    ss = st.session_state
    ss.step_idx = max(ss.step_idx - 1, 0)


def reset():
    ss = st.session_state
    ss.step_idx = 0
