import numpy as np
import streamlit as st
from state import init_state, regenerate, apply_dim_change, go_next, go_prev, reset
from steps import get_step, num_steps
from visualization import flow_diagram, detail_panel, NODE_LAYOUT

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

    new_seq = st.slider("Sequence length", 2, 8, ss.seq_len, key="_seq_len")
    new_dmodel = st.select_slider("d_model", [4, 8, 16, 32], value=ss.d_model, key="_d_model")
    new_dhead = st.select_slider("d_head", [2, 4, 8, 16], value=ss.d_head, key="_d_head")

    dim_changed = (new_seq != ss.seq_len or new_dmodel != ss.d_model or new_dhead != ss.d_head)
    if dim_changed:
        ss.seq_len = new_seq
        ss.d_model = new_dmodel
        ss.d_head = new_dhead
        apply_dim_change()
        st.rerun()

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
        if st.button("Regenerate", use_container_width=True):
            regenerate()
            st.rerun()
    with col_rst:
        if st.button("Reset", use_container_width=True):
            reset()
            st.rerun()

    st.caption(f"Seed: {ss.seed}")

# ---------------------------------------------------------------------------
# Header: step title + navigation
# ---------------------------------------------------------------------------
step  = get_step(ss.step_idx)
total = num_steps()

st.markdown(
    f"<h1 style='font-size:2.4rem;font-weight:800;margin-bottom:0.1rem'>{step['title']}</h1>"
    f"<p style='font-size:1.05rem;color:#666;margin-top:0;margin-bottom:0.8rem'>{step['subtitle']}</p>",
    unsafe_allow_html=True,
)

nav_l, nav_r, nav_prog = st.columns([1, 1, 8])
with nav_l:
    if st.button("← Prev", disabled=(ss.step_idx == 0), use_container_width=True):
        go_prev()
        st.rerun()
with nav_r:
    if st.button("Next →", disabled=(ss.step_idx == total - 1), use_container_width=True):
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
    fig = flow_diagram(
        matrices=ss.matrices,
        active_nodes=step["active_nodes"],
        active_edges=step["active_edges"],
        seq_len=ss.seq_len,
        d_model=ss.d_model,
        d_head=ss.d_head,
        tokens=ss.tokens,
    )
    st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))

with right_col:
    st.markdown("#### Inspect matrix")
    node_options = list(NODE_LAYOUT.keys())

    default_idx = 0
    for i, n in enumerate(node_options):
        if n in step["active_nodes"]:
            default_idx = i
            break

    selected = st.selectbox("Select node", node_options, index=default_idx)
    det_fig = detail_panel(
        node_name=selected,
        matrices=ss.matrices,
        tokens=ss.tokens,
        seq_len=ss.seq_len,
        d_model=ss.d_model,
        d_head=ss.d_head,
    )
    st.plotly_chart(det_fig, use_container_width=True, config=dict(displayModeBar=False))

    mat = ss.matrices.get(selected)
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
