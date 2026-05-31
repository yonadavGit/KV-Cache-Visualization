import streamlit as st
from attention_math import make_matrices, compute_attention
from steps import num_steps


def _recompute():
    ss = st.session_state
    X, Wq, Wk, Wv = make_matrices(ss.seq_len, ss.d_model, ss.d_head, seed=ss.seed)
    Q, K, V, S, S_scaled, S_masked, A, C = compute_attention(X, Wq, Wk, Wv)
    ss.matrices = dict(X=X, Wq=Wq, Wk=Wk, Wv=Wv, Q=Q, K=K, V=V,
                       S=S, S_scaled=S_scaled, S_masked=S_masked, A=A, C=C)


def init_state():
    ss = st.session_state
    if "initialized" not in ss:
        ss.seq_len = 4
        ss.d_model = 8
        ss.d_head = 4
        ss.seed = 42
        ss.step_idx = 0
        ss.tokens = ["The", "cat", "sat", "mat"]
        ss.initialized = True
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


def go_next():
    ss = st.session_state
    ss.step_idx = min(ss.step_idx + 1, num_steps() - 1)


def go_prev():
    ss = st.session_state
    ss.step_idx = max(ss.step_idx - 1, 0)


def reset():
    ss = st.session_state
    ss.step_idx = 0
