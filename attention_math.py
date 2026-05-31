import numpy as np


def make_matrices(seq_len, d_model, d_head, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((seq_len, d_model)).astype(np.float32)
    Wq = rng.standard_normal((d_model, d_head)).astype(np.float32) * 0.3
    Wk = rng.standard_normal((d_model, d_head)).astype(np.float32) * 0.3
    Wv = rng.standard_normal((d_model, d_head)).astype(np.float32) * 0.3
    return X, Wq, Wk, Wv


def compute_attention(X, Wq, Wk, Wv):
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    d_head = Q.shape[1]
    S = Q @ K.T
    S_scaled = S / np.sqrt(d_head)
    seq_len = S.shape[0]
    mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    S_masked = S_scaled.copy()
    S_masked[mask] = -np.inf
    S_shifted = S_masked - np.nanmax(
        np.where(np.isfinite(S_masked), S_masked, -np.inf), axis=1, keepdims=True
    )
    exp_s = np.exp(S_shifted)
    exp_s[mask] = 0.0
    A = exp_s / exp_s.sum(axis=1, keepdims=True)
    C = A @ V
    return Q, K, V, S, S_scaled, S_masked, A, C
