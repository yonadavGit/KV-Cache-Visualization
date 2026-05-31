import numpy as np
import streamlit as st
from state import init_state, regenerate, apply_dim_change, apply_post_change, go_next, go_prev, reset
from steps import get_step, num_steps
from visualization import (
    flow_diagram,
    detail_panel,
    kv_cache_diagram,
    kv_cache_detail_panel,
    multi_head_diagram,
    multi_head_detail_panel,
    KV_CACHE_LAYOUT,
    post_context_diagram,
    post_context_detail_panel,
    MULTI_HEAD_LAYOUT,
    NODE_LAYOUT,
    POST_CONTEXT_LAYOUT,
)

st.set_page_config(
    page_title="Attention Flow Visualizer",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()
ss = st.session_state

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Settings")

    flow_label = st.radio(
        "Flow",
        ["Attention", "Multi-head", "KV cache", "Post-context"],
        index={"attention": 0, "multi_head": 1, "kv_cache": 2, "post_context": 3}.get(ss.flow_mode, 0),
        key="_flow_mode",
    )
    new_flow_mode = {
        "Attention": "attention",
        "Multi-head": "multi_head",
        "KV cache": "kv_cache",
        "Post-context": "post_context",
    }[flow_label]
    if new_flow_mode != ss.flow_mode:
        ss.flow_mode = new_flow_mode
        ss.step_idx = 0
        st.rerun()

    st.divider()

    new_seq = st.slider("Sequence length", 2, 8, ss.seq_len, key="_seq_len")
    new_dmodel = st.select_slider("d_model", [4, 8, 16, 32], value=ss.d_model, key="_d_model")
    new_dhead = st.select_slider("d_head", [2, 4, 8, 16], value=ss.d_head, key="_d_head")
    st.caption(f"Context matrix C: {new_seq} rows × {new_dhead} columns = {new_seq * new_dhead} values")
    if ss.flow_mode == "multi_head":
        st.caption(f"Multi-head demo: 2 heads × {new_dhead} dims → concat width {2 * new_dhead}")
    if ss.flow_mode == "kv_cache":
        st.caption(f"KV cache demo: previous cache has {max(new_seq - 1, 1)} rows; new token adds 1 row")

    dim_changed = (new_seq != ss.seq_len or new_dmodel != ss.d_model or new_dhead != ss.d_head)
    if dim_changed:
        ss.seq_len = new_seq
        ss.d_model = new_dmodel
        ss.d_head = new_dhead
        apply_dim_change()
        st.rerun()

    if ss.flow_mode == "post_context":
        st.divider()
        st.write("**Post-context knobs**")
        new_ffn_mult = st.select_slider(
            "FFN hidden width",
            [1, 2, 4, 8],
            value=ss.ffn_hidden_mult,
            format_func=lambda v: f"{v}× d_head",
            key="_ffn_hidden_mult",
        )
        new_vocab_size = st.select_slider(
            "Toy vocab size",
            [8, 12, 16, 24, 32],
            value=ss.post_vocab_size,
            key="_post_vocab_size",
        )
        post_changed = (
            new_ffn_mult != ss.ffn_hidden_mult
            or new_vocab_size != ss.post_vocab_size
        )
        if post_changed:
            ss.ffn_hidden_mult = new_ffn_mult
            ss.post_vocab_size = new_vocab_size
            apply_post_change()
            st.rerun()

        hidden_dim = ss.d_head * ss.ffn_hidden_mult
        ffn_params = (ss.d_head * hidden_dim) + hidden_dim + (hidden_dim * ss.d_head) + ss.d_head
        out_params = (ss.d_head * ss.post_vocab_size) + ss.post_vocab_size
        st.caption(f"FFN hidden dim: {hidden_dim}")
        st.caption(f"FFN learned params: {ffn_params}")
        st.caption(f"Output projection params: {out_params}")

    st.divider()
    st.write("**Token labels**")
    tokens_input = st.text_input(
        "Comma-separated tokens",
        value=", ".join(ss.tokens),
        key="_tokens_input",
    )
    new_tokens = [t.strip() for t in tokens_input.split(",") if t.strip()]
    if len(new_tokens) >= ss.seq_len:
        ss.tokens = new_tokens[:ss.seq_len]

    st.divider()
    col_r, col_rst = st.columns(2)
    with col_r:
        if st.button("Regenerate", width="stretch"):
            regenerate()
            st.rerun()
    with col_rst:
        if st.button("Reset", width="stretch"):
            reset()
            st.rerun()

    st.caption(f"Seed: {ss.seed}")

# ---------------------------------------------------------------------------
# Header: step title + navigation
# ---------------------------------------------------------------------------
total = num_steps(ss.flow_mode)
if ss.step_idx >= total:
    ss.step_idx = total - 1

step  = get_step(ss.step_idx, ss.flow_mode)

st.markdown(
    f"<h1 style='font-size:2.4rem;font-weight:800;margin-bottom:0.1rem'>{step['title']}</h1>"
    f"<p style='font-size:1.05rem;color:#666;margin-top:0;margin-bottom:0.8rem'>{step['subtitle']}</p>",
    unsafe_allow_html=True,
)

nav_l, nav_r, nav_prog = st.columns([1, 1, 8])
with nav_l:
    if st.button("← Prev", disabled=(ss.step_idx == 0), width="stretch"):
        go_prev()
        st.rerun()
with nav_r:
    if st.button("Next →", disabled=(ss.step_idx == total - 1), width="stretch"):
        go_next()
        st.rerun()
with nav_prog:
    st.progress((ss.step_idx + 1) / total, text=f"Step {ss.step_idx + 1} of {total}")

# Verbal explanation — fill in dynamic values
explanation = step["explanation"]
if "{d_head}" in explanation:
    import math
    explanation = explanation.replace("{d_head}", str(ss.d_head))
    explanation = explanation.replace("{divisor}", f"{math.sqrt(ss.d_head):.2f}")

with st.expander("What's happening here?", expanded=True):
    st.markdown(explanation)

st.divider()

# ---------------------------------------------------------------------------
# Two-column layout: flow diagram | matrix inspector
# ---------------------------------------------------------------------------
left_col, right_col = st.columns([5, 1])

with left_col:
    if ss.flow_mode == "kv_cache":
        fig = kv_cache_diagram(
            matrices=ss.matrices,
            active_nodes=step["active_nodes"],
            active_edges=step["active_edges"],
            seq_len=ss.seq_len,
            d_model=ss.d_model,
            d_head=ss.d_head,
            tokens=ss.tokens,
        )
    elif ss.flow_mode == "multi_head":
        fig = multi_head_diagram(
            matrices=ss.multi_head_matrices,
            active_nodes=step["active_nodes"],
            active_edges=step["active_edges"],
            tokens=ss.tokens,
        )
    elif ss.flow_mode == "post_context":
        fig = post_context_diagram(
            matrices=ss.post_matrices,
            active_nodes=step["active_nodes"],
            active_edges=step["active_edges"],
            tokens=ss.tokens,
        )
    else:
        fig = flow_diagram(
            matrices=ss.matrices,
            active_nodes=step["active_nodes"],
            active_edges=step["active_edges"],
            seq_len=ss.seq_len,
            d_model=ss.d_model,
            d_head=ss.d_head,
            tokens=ss.tokens,
        )
    st.plotly_chart(fig, width="stretch", config=dict(displayModeBar=False))

with right_col:
    st.markdown("#### Inspect matrix")
    if ss.flow_mode == "multi_head":
        node_options = list(MULTI_HEAD_LAYOUT.keys())
    elif ss.flow_mode == "kv_cache":
        node_options = list(KV_CACHE_LAYOUT.keys())
    elif ss.flow_mode == "post_context":
        node_options = list(POST_CONTEXT_LAYOUT.keys())
    else:
        node_options = list(NODE_LAYOUT.keys())

    default_idx = 0
    for i, n in enumerate(node_options):
        if n in step["active_nodes"]:
            default_idx = i
            break

    selected = st.selectbox("Select node", node_options, index=default_idx)
    if ss.flow_mode == "kv_cache":
        det_fig = detail_panel(
            node_name=selected,
            matrices=ss.matrices,
            tokens=ss.tokens,
            seq_len=ss.seq_len,
            d_model=ss.d_model,
            d_head=ss.d_head,
        )
        current_matrices = ss.matrices
    elif ss.flow_mode == "multi_head":
        det_fig = multi_head_detail_panel(
            node_name=selected,
            matrices=ss.multi_head_matrices,
            tokens=ss.tokens,
        )
        current_matrices = ss.multi_head_matrices
    elif ss.flow_mode == "post_context":
        det_fig = post_context_detail_panel(
            node_name=selected,
            matrices=ss.post_matrices,
            tokens=ss.tokens,
        )
        current_matrices = ss.post_matrices
    else:
        det_fig = detail_panel(
            node_name=selected,
            matrices=ss.matrices,
            tokens=ss.tokens,
            seq_len=ss.seq_len,
            d_model=ss.d_model,
            d_head=ss.d_head,
        )
        current_matrices = ss.matrices
    st.plotly_chart(det_fig, width="stretch", config=dict(displayModeBar=False))

    mat = current_matrices.get(selected)
    if mat is not None:
        finite = mat[np.isfinite(mat)]
        st.markdown(f"""
| | value |
|--|--|
| **min** | `{finite.min():.4f}` |
| **max** | `{finite.max():.4f}` |
| **mean** | `{finite.mean():.4f}` |
| **std** | `{finite.std():.4f}` |
""")
        if selected == "A":
            row_sums = mat.sum(axis=1)
            st.caption(f"Row sums: {[f'{v:.3f}' for v in row_sums]}")
        if selected == "Probs":
            st.caption(f"Probability sum: {mat.sum():.3f}")
