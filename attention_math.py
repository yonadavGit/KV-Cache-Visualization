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


def _layer_norm(x, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def _softmax(x):
    shifted = x - x.max(axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


def compute_post_context(C, seed=42, vocab_size=12, ffn_hidden_mult=2):
    rng = np.random.default_rng(seed + 1000)
    d_head = C.shape[1]
    hidden_dim = d_head * ffn_hidden_mult

    W1 = rng.standard_normal((d_head, hidden_dim)).astype(np.float32) * 0.35
    W2 = rng.standard_normal((hidden_dim, d_head)).astype(np.float32) * 0.35
    Wo = rng.standard_normal((d_head, vocab_size)).astype(np.float32) * 0.45

    ffn_hidden = np.maximum(C @ W1, 0)
    ffn_out = ffn_hidden @ W2
    block_out = _layer_norm(C + ffn_out)

    # A compact stand-in for repeatedly passing through more Transformer blocks.
    stack_out = _layer_norm(block_out + 0.25 * np.tanh(block_out))
    final_norm = _layer_norm(stack_out)
    logits = final_norm @ Wo
    last_logits = logits[-1:, :]
    probs = _softmax(last_logits)

    return dict(
        C=C,
        FFNOut=ffn_out,
        BlockOut=block_out,
        StackOut=stack_out,
        FinalNorm=final_norm,
        Logits=logits,
        LastLogits=last_logits,
        Probs=probs,
    )


def compute_multi_head(X, seed=42, num_heads=2, d_head=None):
    rng = np.random.default_rng(seed + 2000)
    seq_len, d_model = X.shape
    d_head = d_head or max(1, d_model // (num_heads * 2))
    combined_dim = num_heads * d_head

    head_outputs = {}
    contexts = []
    for i in range(num_heads):
        Wq = rng.standard_normal((d_model, d_head)).astype(np.float32) * 0.3
        Wk = rng.standard_normal((d_model, d_head)).astype(np.float32) * 0.3
        Wv = rng.standard_normal((d_model, d_head)).astype(np.float32) * 0.3

        Q = X @ Wq
        K = X @ Wk
        V = X @ Wv
        S = (Q @ K.T) / np.sqrt(d_head)
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        S[mask] = -np.inf
        S_shifted = S - np.nanmax(
            np.where(np.isfinite(S), S, -np.inf), axis=1, keepdims=True
        )
        exp_s = np.exp(S_shifted)
        exp_s[mask] = 0.0
        A = exp_s / exp_s.sum(axis=1, keepdims=True)
        C = A @ V

        suffix = str(i + 1)
        head_outputs[f"Wq{suffix}"] = Wq
        head_outputs[f"Wk{suffix}"] = Wk
        head_outputs[f"Wv{suffix}"] = Wv
        head_outputs[f"Q{suffix}"] = Q
        head_outputs[f"K{suffix}"] = K
        head_outputs[f"V{suffix}"] = V
        head_outputs[f"A{suffix}"] = A
        head_outputs[f"C{suffix}"] = C
        contexts.append(C)

    Concat = np.concatenate(contexts, axis=1)
    Wo = rng.standard_normal((combined_dim, d_model)).astype(np.float32) * 0.35
    MHAOut = Concat @ Wo

    return dict(X=X, **head_outputs, Concat=Concat, Wo=Wo, MHAOut=MHAOut)


def compute_kv_cache_demo(X, Wq, Wk, Wv):
    Q, K, V, _, _, _, _, _ = compute_attention(X, Wq, Wk, Wv)
    d_head = Q.shape[1]

    q_new = Q[-1:, :]
    k_new = K[-1:, :]
    v_new = V[-1:, :]
    K_cache = K[:-1, :]
    V_cache = V[:-1, :]
    K_full = K
    V_full = V

    ScoresNew = (q_new @ K_full.T) / np.sqrt(d_head)
    AttnNew = _softmax(ScoresNew)
    ContextNew = AttnNew @ V_full

    return dict(
        XFull=X,
        XNew=X[-1:, :],
        q_new=q_new,
        k_new=k_new,
        v_new=v_new,
        K_cache=K_cache,
        V_cache=V_cache,
        K_full=K_full,
        V_full=V_full,
        ScoresNew=ScoresNew,
        AttnNew=AttnNew,
        ContextNew=ContextNew,
    )


def compute_multi_query_attention(X, seed=42, num_heads=2, d_head=None):
    rng = np.random.default_rng(seed + 3000)
    seq_len, d_model = X.shape
    d_head = d_head or max(1, d_model // (num_heads * 2))
    combined_dim = num_heads * d_head

    Wk = rng.standard_normal((d_model, d_head)).astype(np.float32) * 0.3
    Wv = rng.standard_normal((d_model, d_head)).astype(np.float32) * 0.3
    K = X @ Wk
    V = X @ Wv

    outputs = {"X": X, "Wk": Wk, "Wv": Wv, "K": K, "V": V}
    contexts = []
    mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)

    for i in range(num_heads):
        suffix = str(i + 1)
        Wq = rng.standard_normal((d_model, d_head)).astype(np.float32) * 0.3
        Q = X @ Wq
        S = (Q @ K.T) / np.sqrt(d_head)
        S[mask] = -np.inf
        S_shifted = S - np.nanmax(
            np.where(np.isfinite(S), S, -np.inf), axis=1, keepdims=True
        )
        exp_s = np.exp(S_shifted)
        exp_s[mask] = 0.0
        A = exp_s / exp_s.sum(axis=1, keepdims=True)
        C = A @ V

        outputs[f"Wq{suffix}"] = Wq
        outputs[f"Q{suffix}"] = Q
        outputs[f"A{suffix}"] = A
        outputs[f"C{suffix}"] = C
        contexts.append(C)

    Concat = np.concatenate(contexts, axis=1)
    Wo = rng.standard_normal((combined_dim, d_model)).astype(np.float32) * 0.35
    MQAOut = Concat @ Wo
    outputs["Concat"] = Concat
    outputs["Wo"] = Wo
    outputs["MQAOut"] = MQAOut
    return outputs


def compute_grouped_query_attention(X, seed=42, num_heads=4, num_groups=2, d_head=None):
    rng = np.random.default_rng(seed + 4000)
    seq_len, d_model = X.shape
    d_head = d_head or max(1, d_model // num_heads)
    heads_per_group = num_heads // num_groups
    combined_dim = num_heads * d_head

    outputs = {"X": X}
    group_kv = {}
    for group_idx in range(num_groups):
        suffix = str(group_idx + 1)
        Wk = rng.standard_normal((d_model, d_head)).astype(np.float32) * 0.3
        Wv = rng.standard_normal((d_model, d_head)).astype(np.float32) * 0.3
        K = X @ Wk
        V = X @ Wv
        outputs[f"WkG{suffix}"] = Wk
        outputs[f"WvG{suffix}"] = Wv
        outputs[f"KG{suffix}"] = K
        outputs[f"VG{suffix}"] = V
        group_kv[group_idx] = (K, V)

    contexts = []
    mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    for head_idx in range(num_heads):
        suffix = str(head_idx + 1)
        group_idx = head_idx // heads_per_group
        group_suffix = str(group_idx + 1)
        K, V = group_kv[group_idx]

        Wq = rng.standard_normal((d_model, d_head)).astype(np.float32) * 0.3
        Q = X @ Wq
        S = (Q @ K.T) / np.sqrt(d_head)
        S[mask] = -np.inf
        S_shifted = S - np.nanmax(
            np.where(np.isfinite(S), S, -np.inf), axis=1, keepdims=True
        )
        exp_s = np.exp(S_shifted)
        exp_s[mask] = 0.0
        A = exp_s / exp_s.sum(axis=1, keepdims=True)
        C = A @ V

        outputs[f"Wq{suffix}"] = Wq
        outputs[f"Q{suffix}"] = Q
        outputs[f"A{suffix}"] = A
        outputs[f"C{suffix}"] = C
        outputs[f"head{suffix}_group"] = group_suffix
        contexts.append(C)

    Concat = np.concatenate(contexts, axis=1)
    Wo = rng.standard_normal((combined_dim, d_model)).astype(np.float32) * 0.35
    GQAOut = Concat @ Wo
    outputs["Concat"] = Concat
    outputs["Wo"] = Wo
    outputs["GQAOut"] = GQAOut
    return outputs


def compute_mla_attention(X, seed=42, d_head=None, d_latent=None):
    rng = np.random.default_rng(seed + 5000)
    seq_len, d_model = X.shape
    d_head = d_head or max(1, d_model // 2)
    d_latent = d_latent or max(2, d_head // 2)

    Wq = rng.standard_normal((d_model, d_head)).astype(np.float32) * 0.3
    Wdkv = rng.standard_normal((d_model, d_latent)).astype(np.float32) * 0.3
    Wuk = rng.standard_normal((d_latent, d_head)).astype(np.float32) * 0.3
    Wuv = rng.standard_normal((d_latent, d_head)).astype(np.float32) * 0.3
    Wo = rng.standard_normal((d_head, d_model)).astype(np.float32) * 0.35

    Q = X @ Wq
    cKV = X @ Wdkv
    K = cKV @ Wuk
    V = cKV @ Wuv

    S = Q @ K.T
    S_scaled = S / np.sqrt(d_head)
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
    MLAOut = C @ Wo

    Wscore = Wq @ Wuk.T
    Wout = Wuv @ Wo
    XNew = X[-1:, :]
    S_abs = XNew @ Wscore @ cKV.T
    S_abs_scaled = S_abs / np.sqrt(d_head)
    A_abs = _softmax(S_abs_scaled)
    MLAOut_abs = A_abs @ cKV @ Wout

    return dict(
        X=X,
        XNew=XNew,
        Wq=Wq,
        Q=Q,
        Wdkv=Wdkv,
        cKV=cKV,
        Wuk=Wuk,
        Wuv=Wuv,
        Wo=Wo,
        Wscore=Wscore,
        Wout=Wout,
        K=K,
        V=V,
        S=S,
        S_scaled=S_scaled,
        S_masked=S_masked,
        A=A,
        C=C,
        MLAOut=MLAOut,
        S_abs=S_abs,
        A_abs=A_abs,
        MLAOut_abs=MLAOut_abs,
    )
