import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Layout (paper coords 0–1)
#
# Three horizontal rows:
#   TOP  y=0.80  — Wq, Q
#   MID  y=0.50  — X, Wk, K,  S, S_scaled, S_masked, A,  C
#   BOT  y=0.20  — Wv, V
#
# Benefit: K→S is a straight horizontal arrow; Q→S curves from top;
#           V→C curves from bottom (passes below score matrices safely).
# ---------------------------------------------------------------------------
_NW = 0.068   # main node width
_NH = 0.115   # main node height
_SW = 0.055   # score-matrix node width
_SH = 0.105   # score-matrix node height

# Layout:  X ──► Wq/Wk/Wv ──► Q/K/V ──► score pipeline ──► C
# This keeps the projection multiplication visually left-to-right:
# X is the left operand, W is immediately to its right, and Q/K/V are outputs.
NODE_LAYOUT = {
    #         cx      cy     w    h     label_side   node_type
    "X":        dict(cx=0.055, cy=0.50, w=_NW, h=_NH, ls="above", nt="input"),
    "Wq":       dict(cx=0.245, cy=0.80, w=_NW, h=_NH, ls="above", nt="param"),
    "Wk":       dict(cx=0.245, cy=0.50, w=_NW, h=_NH, ls="above", nt="param"),
    "Wv":       dict(cx=0.245, cy=0.20, w=_NW, h=_NH, ls="above", nt="param"),
    "Q":        dict(cx=0.370, cy=0.80, w=_NW, h=_NH, ls="above", nt="activation"),
    "K":        dict(cx=0.378, cy=0.50, w=_NW, h=_NH, ls="above", nt="activation"),
    "V":        dict(cx=0.378, cy=0.20, w=_NW, h=_NH, ls="above", nt="activation"),
    "S":        dict(cx=0.470, cy=0.50, w=_SW, h=_SH, ls="above", nt="score"),
    "S_scaled": dict(cx=0.568, cy=0.50, w=_SW, h=_SH, ls="above", nt="score"),
    "S_masked": dict(cx=0.666, cy=0.50, w=_SW, h=_SH, ls="above", nt="score"),
    "A":        dict(cx=0.764, cy=0.50, w=_SW, h=_SH, ls="above", nt="attn"),
    "C":        dict(cx=0.940, cy=0.50, w=_NW, h=_NH, ls="above", nt="output"),
}

NODE_ROLES = {
    "X":        ("Input Embeddings",  ""),
    "Wq":       ("Query Weight",      "learned param"),
    "Wk":       ("Key Weight",        "learned param"),
    "Wv":       ("Value Weight",      "learned param"),
    "Q":        ("Queries",           "= X · Wq"),
    "K":        ("Keys",              "= X · Wk"),
    "V":        ("Values",            "= X · Wv"),
    "S":        ("Raw Scores",        "= Q · Kᵀ"),
    "S_scaled": ("Scaled Scores",     "= S / √d"),
    "S_masked": ("Masked Scores",     "causal mask"),
    "A":        ("Attn Weights",      "= softmax(·)"),
    "C":        ("Context Output",    "= A · V"),
}

# Color by node type
_TYPE_COLOR = {
    "input":      "#7d3c98",   # purple  — input X
    "param":      "#b7770d",   # amber   — learned weights Wq/Wk/Wv
    "activation": "#c0392b",   # red     — Q, K, V
    "score":      "#1a5276",   # deep blue — S matrices
    "attn":       "#117a65",   # teal    — A
    "output":     "#a04000",   # burnt orange — C
}

NODE_COLOR = {n: _TYPE_COLOR[v["nt"]] for n, v in NODE_LAYOUT.items()}

PROJECTION_EDGES = [
    dict(weight="Wq", out="Q", logical={"X->Q", "Wq->Q"}, label="X · Wq"),
    dict(weight="Wk", out="K", logical={"X->K", "Wk->K"}, label="X · Wk"),
    dict(weight="Wv", out="V", logical={"X->V", "Wv->V"}, label="X · Wv"),
]

# route:
#   "scurve" — S-curve, arrives from left  (arrowhead: ax=-16, ay=0)
#   "above"  — arc over obstacles, arrives from above (ax=0, ay=-18)
#   "below"  — arc under obstacles, arrives from below (ax=0, ay=+18)
#   "short"  — short horizontal
EDGES = [
    # Score matmul: Q and K converge on S
    ("Q",  "S",  "Q · Kᵀ", 0.50, 0.78, True,  "scurve"),
    ("K",  "S",  "",        0.50, 0.22, False, "scurve"),
    # Score pipeline
    ("S",        "S_scaled", "÷ √d_head|element-wise scalar",        0.50, 0.50, False, "scurve"),
    ("S_scaled", "S_masked", "causal mask|upper-tri → −∞",           0.50, 0.50, False, "scurve"),
    ("S_masked", "A",        "softmax|row-wise → each row sums to 1",0.50, 0.50, False, "scurve"),
    # Context matmul: A and V converge on C
    ("A",  "C",  "A · V",  0.50, 0.72, True,  "scurve"),
    ("V",  "C",  "",       0.50, 0.28, False, "scurve"),
]

ACTIVE_ALPHA   = 1.0
INACTIVE_ALPHA = 0.15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = h[0]*2 + h[1]*2 + h[2]*2
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


def _node_bounds(name):
    n = NODE_LAYOUT[name]
    hw, hh = n["w"] / 2, n["h"] / 2
    return n["cx"] - hw, n["cy"] - hh, n["cx"] + hw, n["cy"] + hh


def _port(name, side, y_frac=0.5):
    x0, y0, x1, y1 = _node_bounds(name)
    py = y0 + y_frac * (y1 - y0)
    return (x1, py) if side == "right" else (x0, py)


def _bezier_path(sx, sy, dx, dy, route="scurve"):
    if route == "scurve" or route == "short":
        mid = (sx + dx) / 2
        return f"M {sx:.4f} {sy:.4f} C {mid:.4f} {sy:.4f} {mid:.4f} {dy:.4f} {dx:.4f} {dy:.4f}"
    elif route == "above":
        apex = max(sy, dy) + 0.14
        return f"M {sx:.4f} {sy:.4f} C {sx:.4f} {apex:.4f} {dx:.4f} {apex:.4f} {dx:.4f} {dy:.4f}"
    elif route == "below":
        nadir = min(sy, dy) - 0.09
        return f"M {sx:.4f} {sy:.4f} C {sx:.4f} {nadir:.4f} {dx:.4f} {nadir:.4f} {dx:.4f} {dy:.4f}"
    elif route == "fan":
        # Stay at start-x while reaching dest-y, then sweep right — arrives from LEFT.
        # Used for X→op connections: avoids W nodes, arrow clearly comes from left.
        return f"M {sx:.4f} {sy:.4f} C {sx:.4f} {dy:.4f} {dx:.4f} {dy:.4f} {dx:.4f} {dy:.4f}"
    elif route == "vert":
        # Straight vertical line — used for W→op (W is directly above op, same cx).
        return f"M {sx:.4f} {sy:.4f} L {dx:.4f} {dy:.4f}"
    mid = (sx + dx) / 2
    return f"M {sx:.4f} {sy:.4f} C {mid:.4f} {sy:.4f} {mid:.4f} {dy:.4f} {dx:.4f} {dy:.4f}"


def _arrowhead_offset(route):
    """Return (ax_px, ay_px) for annotation arrowhead tail offset."""
    if route == "above":
        return 0, -18
    elif route == "below":
        return 0, 18
    elif route == "vert":
        return 0, -18   # arrives from above (W above op)
    return -16, 0       # arrives from left (scurve/short/fan)


def _shape_label(name, seq_len, d_model, d_head):
    m = {
        "X": f"{seq_len}×{d_model}",
        "Wq": f"{d_model}×{d_head}", "Wk": f"{d_model}×{d_head}", "Wv": f"{d_model}×{d_head}",
        "Q": f"{seq_len}×{d_head}", "K": f"{seq_len}×{d_head}", "V": f"{seq_len}×{d_head}",
        "S": f"{seq_len}×{seq_len}", "S_scaled": f"{seq_len}×{seq_len}",
        "S_masked": f"{seq_len}×{seq_len}", "A": f"{seq_len}×{seq_len}",
        "C": f"{seq_len}×{d_head}",
    }
    return m.get(name, "")


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def flow_diagram(matrices, active_nodes, active_edges,
                 seq_len, d_model, d_head, tokens,
                 fig_width=1400, fig_height=700):

    active_node_set = set(active_nodes)
    active_edge_set = set(active_edges)

    shapes      = []
    annotations = []
    traces      = []

    # ---------------------------------------------------------------
    # 1. Node boxes (layer=below so heatmap renders on top of fill)
    # ---------------------------------------------------------------
    for node_name, layout in NODE_LAYOUT.items():
        is_active  = node_name in active_node_set
        color      = NODE_COLOR[node_name]
        is_param   = layout["nt"] == "param"
        label_above = layout["ls"] == "above"

        border_alpha = 1.0 if is_active else 0.18
        fill_alpha   = 0.13 if is_active else 0.04
        border_w     = 2.5 if is_active else 1.0
        dash_style   = "dash" if is_param else "solid"

        x0, y0, x1, y1 = _node_bounds(node_name)
        cx = layout["cx"]

        shapes.append(dict(
            type="rect", xref="paper", yref="paper",
            x0=x0, y0=y0, x1=x1, y1=y1,
            fillcolor=_rgba(color, fill_alpha),
            line=dict(color=_rgba(color, border_alpha), width=border_w, dash=dash_style),
            layer="below",
        ))

        # Node name + role  (anchored so text grows away from the box)
        name_color = color if is_active else _rgba(color, 0.30)
        role_color  = "#333" if is_active else "#ccc"
        role, formula = NODE_ROLES[node_name]

        if label_above:
            label_y, label_yanchor = y1 + 0.016, "bottom"
        else:
            label_y, label_yanchor = y0 - 0.016, "top"

        annotations.append(dict(
            xref="paper", yref="paper",
            x=cx, y=label_y, yanchor=label_yanchor,
            text=(f"<span style='font-size:19px;font-weight:800;color:{name_color}'>{node_name}</span>"
                  f"<br><span style='font-size:12px;color:{role_color}'>{role}</span>"),
            showarrow=False, align="center",
        ))

        # Shape + formula (opposite side from name, also anchored away from box)
        sl = _shape_label(node_name, seq_len, d_model, d_head)
        shape_color   = "#333" if is_active else "#bbb"
        formula_color = "#666" if is_active else "#ddd"

        if label_above:
            shape_y, shape_yanchor = y0 - 0.014, "top"
        else:
            shape_y, shape_yanchor = y1 + 0.014, "bottom"

        annotations.append(dict(
            xref="paper", yref="paper",
            x=cx, y=shape_y, yanchor=shape_yanchor,
            text=(f"<span style='font-size:11px;color:{shape_color}'>[{sl}]</span>"
                  + (f"<br><span style='font-size:10px;color:{formula_color}'>{formula}</span>"
                     if formula else "")),
            showarrow=False, align="center",
        ))

    # ---------------------------------------------------------------
    # 2. Edges (layer=above so they draw over heatmaps)
    # ---------------------------------------------------------------
    for src, dst, label, src_yf, dst_yf, is_matmul, route in EDGES:
        edge_key  = f"{src}->{dst}"
        is_active = edge_key in active_edge_set

        alpha  = 0.85 if is_active else 0.10
        width  = 2.0  if is_active else 0.7
        color  = "#1c2833" if is_active else "#aaa"

        sx, sy = _port(src, "right", src_yf)
        dx, dy = _port(dst, "left",  dst_yf)
        path   = _bezier_path(sx, sy, dx, dy, route)

        shapes.append(dict(
            type="path", xref="paper", yref="paper",
            path=path,
            fillcolor="rgba(0,0,0,0)",
            line=dict(color=_rgba(color, alpha), width=width),
            layer="above",
        ))

        ax_px, ay_px = _arrowhead_offset(route)
        annotations.append(dict(
            xref="paper", yref="paper",
            x=dx, y=dy,
            ax=ax_px, ay=ay_px, axref="pixel", ayref="pixel",
            showarrow=True,
            arrowhead=3, arrowsize=0.9, arrowwidth=width,
            arrowcolor=_rgba(color, alpha),
            text="",
        ))

        # Edge label (active only) — place near the arc midpoint
        if label and is_active:
            if route == "above":
                lx = (sx + dx) / 2
                ly = max(sy, dy) + 0.16   # above the arc apex
            elif route == "below":
                lx = (sx + dx) / 2
                ly = min(sy, dy) - 0.11
            else:
                lx = (sx + dx) / 2
                ly = max(sy, dy) + 0.032
            parts = label.split("|")
            main_text = f"<b>{parts[0]}</b>"
            if len(parts) > 1:
                main_text += f"<br><i style='font-size:9px;color:#555'>{parts[1]}</i>"
            annotations.append(dict(
                xref="paper", yref="paper",
                x=lx, y=ly,
                text=main_text,
                showarrow=False,
                font=dict(size=11, color="#1c2833"),
                bgcolor="rgba(255,255,255,0.92)",
                borderpad=3,
            ))

    # ---------------------------------------------------------------
    # 3. Projection matmuls: X -> Wq/Wk/Wv -> Q/K/V
    # ---------------------------------------------------------------
    for proj in PROJECTION_EDGES:
        is_active = bool(proj["logical"] & active_edge_set)
        alpha = 0.90 if is_active else 0.10
        width = 2.0 if is_active else 0.7
        color = "#1c2833"

        weight_node = proj["weight"]
        out_node = proj["out"]

        sx, sy = _port("X", "right", 0.50)
        dx, dy = _port(weight_node, "left", 0.50)
        path = _bezier_path(sx, sy, dx, dy, "fan")
        shapes.append(dict(
            type="path", xref="paper", yref="paper", path=path,
            fillcolor="rgba(0,0,0,0)",
            line=dict(color=_rgba(color, alpha), width=width),
            layer="above",
        ))
        annotations.append(dict(
            xref="paper", yref="paper", x=dx, y=dy,
            ax=-16, ay=0, axref="pixel", ayref="pixel",
            showarrow=True, arrowhead=3, arrowsize=0.9, arrowwidth=width,
            arrowcolor=_rgba(color, alpha), text="",
        ))

        sx2, sy2 = _port(weight_node, "right", 0.50)
        dx2, dy2 = _port(out_node, "left", 0.50)
        path2 = _bezier_path(sx2, sy2, dx2, dy2, "short")
        shapes.append(dict(
            type="path", xref="paper", yref="paper", path=path2,
            fillcolor="rgba(0,0,0,0)",
            line=dict(color=_rgba(color, alpha), width=width),
            layer="above",
        ))
        annotations.append(dict(
            xref="paper", yref="paper", x=dx2, y=dy2,
            ax=-16, ay=0, axref="pixel", ayref="pixel",
            showarrow=True, arrowhead=3, arrowsize=0.9, arrowwidth=width,
            arrowcolor=_rgba(color, alpha), text="",
        ))

        if is_active:
            annotations.append(dict(
                xref="paper", yref="paper",
                x=(sx2 + dx2) / 2,
                y=max(sy2, dy2) + 0.035,
                text=f"<b>{proj['label']}</b>",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.88)",
                borderpad=2,
                font=dict(size=10, color="#1c2833"),
            ))

    # ---------------------------------------------------------------
    # 4. Heatmap traces
    # ---------------------------------------------------------------
    node_list = list(NODE_LAYOUT.keys())
    for node_name, layout in NODE_LAYOUT.items():
        is_active = node_name in active_node_set
        mat       = matrices.get(node_name)
        if mat is None:
            continue

        display = mat.copy().astype(float)
        if node_name == "S_masked":
            display[~np.isfinite(display)] = np.nan

        rows, cols = display.shape
        use_div = node_name in ("S", "S_scaled", "S_masked")
        is_param_node = layout["nt"] == "param"
        colorscale = "RdBu_r" if use_div else ("YlOrBr" if is_param_node else "Viridis")

        hover = []
        for r in range(rows):
            row_h = []
            tok_r = tokens[r] if r < len(tokens) else str(r)
            for c in range(cols):
                v    = display[r, c]
                tok_c = tokens[c] if c < len(tokens) else str(c)
                if use_div:
                    cell = (f"<b>{node_name}[{tok_r}→{tok_c}]</b><br>"
                            + ("−∞ (masked)" if np.isnan(v) else f"{v:.4f}"))
                else:
                    cell = f"<b>{node_name}[{tok_r}, dim {c}]</b><br>{v:.4f}"
                row_h.append(cell)
            hover.append(row_h)

        if use_div:
            fv = display[np.isfinite(display)]
            vmax = max(abs(fv.max()), abs(fv.min())) if len(fv) else 1.0
            zmin, zmax = -vmax, vmax
        else:
            zmin, zmax = None, None

        idx = node_list.index(node_name)
        xax = "x"     if idx == 0 else f"x{idx+1}"
        yax = "y"     if idx == 0 else f"y{idx+1}"
        xak = "xaxis" if idx == 0 else f"xaxis{idx+1}"
        yak = "yaxis" if idx == 0 else f"yaxis{idx+1}"

        x0, y0, x1, y1 = _node_bounds(node_name)

        # For X: pass explicit y-labels (tokens) so the axis can show them
        row_labels = [tokens[r] if r < len(tokens) else str(r) for r in range(rows)]
        trace_y = row_labels if node_name == "X" else None

        trace = go.Heatmap(
            z=display, colorscale=colorscale,
            zmin=zmin, zmax=zmax,
            showscale=False,
            opacity=0.90 if is_active else 0.18,
            hoverinfo="text",
            hovertemplate="%{text}<extra></extra>",
            text=hover,
            y=trace_y,
            xaxis=xax, yaxis=yax,
        )
        traces.append((trace, xak, yak, x0, y0, x1, y1, node_name))

    # ---------------------------------------------------------------
    # 5. Assemble
    # ---------------------------------------------------------------
    layout_args = dict(
        paper_bgcolor="white", plot_bgcolor="white",
        width=fig_width, height=fig_height,
        margin=dict(l=8, r=8, t=44, b=16),
        showlegend=False,
        shapes=shapes, annotations=annotations,
    )
    for trace, xak, yak, x0, y0, x1, y1, nname in traces:
        if nname == "X":
            # Show token names on X's y-axis (right side = inside the figure)
            layout_args[xak] = dict(
                domain=[x0, x1], showticklabels=False,
                showgrid=False, zeroline=False, fixedrange=True,
            )
            layout_args[yak] = dict(
                domain=[y0, y1],
                showticklabels=True,
                tickfont=dict(size=8, color="#444"),
                side="right",
                showgrid=False, zeroline=False, fixedrange=True,
                autorange="reversed",
            )
        else:
            layout_args[xak] = dict(
                domain=[x0, x1], showticklabels=False,
                showgrid=False, zeroline=False, fixedrange=True,
            )
            layout_args[yak] = dict(
                domain=[y0, y1], showticklabels=False,
                showgrid=False, zeroline=False, fixedrange=True,
            )

    # Direction indicators on X
    x0_X, y0_X, x1_X, y1_X = _node_bounds("X")
    cx_X = NODE_LAYOUT["X"]["cx"]
    # "↑ tokens" rotated label to the left of X
    annotations.append(dict(
        xref="paper", yref="paper",
        x=x0_X - 0.018, y=(y0_X + y1_X) / 2,
        text="tokens ↕",
        showarrow=False,
        textangle=-90,
        font=dict(size=9, color="#666"),
    ))
    # "← d_model dims →" below X (above the shape label)
    annotations.append(dict(
        xref="paper", yref="paper",
        x=cx_X, y=y0_X - 0.046,
        text="← d_model →",
        showarrow=False,
        font=dict(size=9, color="#666"),
    ))

    return go.Figure(data=[t[0] for t in traces], layout=go.Layout(**layout_args))


# ---------------------------------------------------------------------------
# Detail panel
# ---------------------------------------------------------------------------

def detail_panel(node_name, matrices, tokens, seq_len, d_model, d_head):
    mat = matrices.get(node_name)
    if mat is None:
        return go.Figure()

    display = mat.copy().astype(float)
    if node_name == "S_masked":
        display[~np.isfinite(display)] = np.nan

    rows, cols = display.shape
    use_div    = node_name in ("S", "S_scaled", "S_masked")
    is_param_node = NODE_LAYOUT[node_name]["nt"] == "param"
    colorscale = "RdBu_r" if use_div else ("YlOrBr" if is_param_node else "Viridis")

    row_labels = [tokens[r] if r < len(tokens) else str(r) for r in range(rows)]
    col_labels = [tokens[c] if c < len(tokens) else str(c) for c in range(cols)]

    hover = [[("−∞" if np.isnan(display[r, c]) else f"{display[r,c]:.4f}")
              for c in range(cols)] for r in range(rows)]

    if use_div:
        fv = display[np.isfinite(display)]
        vmax = max(abs(fv.max()), abs(fv.min())) if len(fv) else 1.0
        zmin, zmax = -vmax, vmax
    else:
        zmin, zmax = None, None

    role, formula = NODE_ROLES.get(node_name, ("", ""))
    sl  = _shape_label(node_name, seq_len, d_model, d_head)
    is_param = NODE_LAYOUT[node_name]["nt"] == "param"
    param_tag = " 〈learned〉" if is_param else ""
    title_text = f"<b>{node_name}</b>  <span style='color:#888'>{role}{param_tag}  [{sl}]</span>"
    if formula:
        title_text += f"   <span style='color:#aaa'>{formula}</span>"

    fig = go.Figure(go.Heatmap(
        z=display, colorscale=colorscale, zmin=zmin, zmax=zmax,
        showscale=True,
        hovertemplate="%{text}<extra></extra>",
        text=hover, x=col_labels, y=row_labels,
    ))
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=12)),
        height=270,
        margin=dict(l=50, r=10, t=48, b=30),
        paper_bgcolor="white", plot_bgcolor="white",
        yaxis=dict(autorange="reversed"),
    )
    return fig


# ---------------------------------------------------------------------------
# Post-context flow
# ---------------------------------------------------------------------------

POST_CONTEXT_LAYOUT = {
    "C":          dict(cx=0.045, cy=0.52, w=0.062, h=0.145, nt="context"),
    "MLP":        dict(cx=0.155, cy=0.52, w=0.075, h=0.115, nt="module", param=True),
    "FFNOut":     dict(cx=0.265, cy=0.52, w=0.062, h=0.145, nt="ffn"),
    "BlockOut":   dict(cx=0.375, cy=0.52, w=0.062, h=0.145, nt="block"),
    "StackOut":   dict(cx=0.485, cy=0.52, w=0.062, h=0.145, nt="block", param=True),
    "FinalNorm":  dict(cx=0.595, cy=0.52, w=0.062, h=0.145, nt="norm", param=True),
    "Logits":     dict(cx=0.710, cy=0.52, w=0.074, h=0.145, nt="logits", param=True),
    "LastLogits": dict(cx=0.840, cy=0.52, w=0.080, h=0.075, nt="last"),
    "Probs":      dict(cx=0.950, cy=0.52, w=0.070, h=0.075, nt="probs"),
}

POST_CONTEXT_ROLES = {
    "C":          ("Context Matrix", "from attention"),
    "MLP":        ("Feed-Forward Network", "learned MLP weights"),
    "FFNOut":     ("FFN Output", "same shape as C"),
    "BlockOut":   ("Block Output", "residual + norm"),
    "StackOut":   ("Block Stack", "more learned blocks"),
    "FinalNorm":  ("Final Norm", "learned scale/shift"),
    "Logits":     ("Vocabulary Logits", "learned output projection"),
    "LastLogits": ("Last Row", "used for next token"),
    "Probs":      ("Probabilities", "softmax"),
}

_POST_CONTEXT_COLOR = {
    "context": "#a04000",
    "module": "#6c3483",
    "ffn": "#7d3c98",
    "block": "#1a5276",
    "norm": "#117a65",
    "logits": "#b7770d",
    "last": "#c0392b",
    "probs": "#2e86c1",
}

POST_CONTEXT_EDGES = [
    ("C", "MLP", "C->MLP", "into FFN"),
    ("MLP", "FFNOut", "MLP->FFNOut", "MLP output"),
    ("FFNOut", "BlockOut", "FFNOut->BlockOut", "residual + norm"),
    ("BlockOut", "StackOut", "BlockOut->StackOut", "repeat blocks"),
    ("StackOut", "FinalNorm", "StackOut->FinalNorm", "final norm"),
    ("FinalNorm", "Logits", "FinalNorm->Logits", "output layer"),
    ("Logits", "LastLogits", "Logits->LastLogits", "keep last row"),
    ("LastLogits", "Probs", "LastLogits->Probs", "softmax"),
]


def _post_bounds(name):
    n = POST_CONTEXT_LAYOUT[name]
    hw, hh = n["w"] / 2, n["h"] / 2
    return n["cx"] - hw, n["cy"] - hh, n["cx"] + hw, n["cy"] + hh


def _post_port(name, side):
    x0, y0, x1, y1 = _post_bounds(name)
    py = (y0 + y1) / 2
    return (x1, py) if side == "right" else (x0, py)


def _post_shape_label(name, mat):
    rows, cols = mat.shape
    return f"{rows}×{cols}"


def post_context_diagram(matrices, active_nodes, active_edges, tokens,
                         fig_width=1400, fig_height=460):
    active_node_set = set(active_nodes)
    active_edge_set = set(active_edges)
    shapes = []
    annotations = []
    traces = []

    for name, layout in POST_CONTEXT_LAYOUT.items():
        is_active = name in active_node_set
        color = _POST_CONTEXT_COLOR[layout["nt"]]
        has_params = layout.get("param", False)
        x0, y0, x1, y1 = _post_bounds(name)

        shapes.append(dict(
            type="rect", xref="paper", yref="paper",
            x0=x0, y0=y0, x1=x1, y1=y1,
            fillcolor=_rgba(color, 0.12 if is_active else 0.035),
            line=dict(color=_rgba(color, 1.0 if is_active else 0.18),
                      width=2.4 if is_active else 1.0,
                      dash="dash" if has_params else "solid"),
            layer="below",
        ))

        role, formula = POST_CONTEXT_ROLES[name]
        annotations.append(dict(
            xref="paper", yref="paper", x=layout["cx"], y=y1 + 0.030,
            yanchor="bottom", showarrow=False, align="center",
            text=(f"<span style='font-size:18px;font-weight:800;color:{color if is_active else _rgba(color, 0.30)}'>{name}</span>"
                  f"<br><span style='font-size:12px;color:{'#333' if is_active else '#ccc'}'>{role}</span>"),
        ))

        mat = matrices.get(name)
        if mat is not None:
            annotations.append(dict(
                xref="paper", yref="paper", x=layout["cx"], y=y0 - 0.026,
                yanchor="top", showarrow=False, align="center",
                text=(f"<span style='font-size:11px;color:{'#333' if is_active else '#bbb'}'>[{_post_shape_label(name, mat)}]</span>"
                      f"<br><span style='font-size:10px;color:{'#666' if is_active else '#ddd'}'>{formula}</span>"),
            ))
        elif layout["nt"] == "module":
            annotations.append(dict(
                xref="paper", yref="paper", x=layout["cx"], y=layout["cy"],
                showarrow=False, align="center",
                text=("<span style='font-size:11px;font-weight:700'>Linear</span>"
                      "<br><span style='font-size:10px'>activation</span>"
                      "<br><span style='font-size:11px;font-weight:700'>Linear</span>"),
                font=dict(color=_rgba(color, 1.0 if is_active else 0.28)),
            ))

        if has_params:
            annotations.append(dict(
                xref="paper", yref="paper", x=layout["cx"], y=y0 - 0.052,
                yanchor="top", showarrow=False, align="center",
                text=f"<span style='font-size:9px;color:{color if is_active else _rgba(color, 0.25)}'>learned params</span>",
            ))

    for src, dst, edge_key, label in POST_CONTEXT_EDGES:
        is_active = edge_key in active_edge_set
        alpha = 0.86 if is_active else 0.10
        width = 2.0 if is_active else 0.7
        sx, sy = _post_port(src, "right")
        dx, dy = _post_port(dst, "left")
        if edge_key == "C->MLP" and "C" in matrices:
            rows = matrices["C"].shape[0]
            x0_c, y0_c, x1_c, y1_c = _post_bounds("C")
            lane_count = min(rows, 8)
            for r in range(lane_count):
                row_y = y0_c + ((r + 0.5) / rows) * (y1_c - y0_c)
                path = _bezier_path(x1_c, row_y, dx, row_y, "short")
                shapes.append(dict(
                    type="path", xref="paper", yref="paper", path=path,
                    fillcolor="rgba(0,0,0,0)",
                    line=dict(color=_rgba("#1c2833", alpha), width=max(width - 0.2, 0.6)),
                    layer="above",
                ))
                annotations.append(dict(
                    xref="paper", yref="paper", x=dx, y=row_y,
                    ax=-16, ay=0, axref="pixel", ayref="pixel",
                    showarrow=True, arrowhead=3, arrowsize=0.75, arrowwidth=max(width - 0.2, 0.6),
                    arrowcolor=_rgba("#1c2833", alpha), text="",
                ))
        else:
            path = _bezier_path(sx, sy, dx, dy, "short")
            shapes.append(dict(
                type="path", xref="paper", yref="paper", path=path,
                fillcolor="rgba(0,0,0,0)",
                line=dict(color=_rgba("#1c2833", alpha), width=width),
                layer="above",
            ))
            annotations.append(dict(
                xref="paper", yref="paper", x=dx, y=dy,
                ax=-16, ay=0, axref="pixel", ayref="pixel",
                showarrow=True, arrowhead=3, arrowsize=0.9, arrowwidth=width,
                arrowcolor=_rgba("#1c2833", alpha), text="",
            ))
        if is_active:
            annotations.append(dict(
                xref="paper", yref="paper", x=(sx + dx) / 2, y=sy + 0.052,
                text=f"<b>{label}</b>", showarrow=False,
                bgcolor="rgba(255,255,255,0.90)", borderpad=2,
                font=dict(size=10, color="#1c2833"),
            ))
            if edge_key == "C->MLP":
                annotations.append(dict(
                    xref="paper", yref="paper", x=(sx + dx) / 2, y=sy - 0.074,
                    text="<span style='font-size:10px'>same MLP reused for every row</span>",
                    showarrow=False,
                    bgcolor="rgba(255,255,255,0.90)",
                    borderpad=2,
                    font=dict(color="#1c2833"),
                ))

    node_list = list(POST_CONTEXT_LAYOUT.keys())
    for name, layout in POST_CONTEXT_LAYOUT.items():
        mat = matrices.get(name)
        if mat is None:
            continue
        is_active = name in active_node_set
        display = mat.copy().astype(float)
        rows, cols = display.shape
        hover = [[f"<b>{name}[{r},{c}]</b><br>{display[r, c]:.4f}"
                  for c in range(cols)] for r in range(rows)]

        idx = node_list.index(name)
        xax = "x" if idx == 0 else f"x{idx+1}"
        yax = "y" if idx == 0 else f"y{idx+1}"
        xak = "xaxis" if idx == 0 else f"xaxis{idx+1}"
        yak = "yaxis" if idx == 0 else f"yaxis{idx+1}"
        x0, y0, x1, y1 = _post_bounds(name)

        row_labels = [tokens[r] if r < len(tokens) else str(r) for r in range(rows)]
        if name in ("LastLogits", "Probs"):
            row_labels = ["last"]
        col_labels = [f"v{c}" for c in range(cols)]

        trace = go.Heatmap(
            z=display,
            colorscale="Blues" if name == "Probs" else "Viridis",
            showscale=False,
            opacity=0.92 if is_active else 0.18,
            hoverinfo="text",
            hovertemplate="%{text}<extra></extra>",
            text=hover,
            x=col_labels,
            y=row_labels,
            xaxis=xax,
            yaxis=yax,
        )
        traces.append((trace, xak, yak, x0, y0, x1, y1))

    layout_args = dict(
        paper_bgcolor="white", plot_bgcolor="white",
        width=fig_width, height=fig_height,
        margin=dict(l=8, r=8, t=70, b=58),
        showlegend=False,
        shapes=shapes,
        annotations=annotations,
    )

    for _, xak, yak, x0, y0, x1, y1 in traces:
        layout_args[xak] = dict(
            domain=[x0, x1], showticklabels=False,
            showgrid=False, zeroline=False, fixedrange=True,
        )
        layout_args[yak] = dict(
            domain=[y0, y1], showticklabels=False,
            showgrid=False, zeroline=False, fixedrange=True,
            autorange="reversed",
        )

    return go.Figure(data=[t[0] for t in traces], layout=go.Layout(**layout_args))


def post_context_detail_panel(node_name, matrices, tokens):
    mat = matrices.get(node_name)
    if mat is None:
        role, formula = POST_CONTEXT_ROLES.get(node_name, ("", ""))
        fig = go.Figure()
        fig.update_layout(
            title=dict(
                text=f"<b>{node_name}</b>  <span style='color:#888'>{role}</span>",
                font=dict(size=12),
            ),
            height=270,
            margin=dict(l=20, r=20, t=48, b=20),
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[
                dict(
                    x=0.5,
                    y=0.58,
                    xref="paper",
                    yref="paper",
                    text="Linear → activation → Linear",
                    showarrow=False,
                    font=dict(size=16, color="#333"),
                ),
                dict(
                    x=0.5,
                    y=0.42,
                    xref="paper",
                    yref="paper",
                    text=formula,
                    showarrow=False,
                    font=dict(size=12, color="#777"),
                ),
            ],
        )
        return fig

    display = mat.copy().astype(float)
    rows, cols = display.shape
    role, formula = POST_CONTEXT_ROLES.get(node_name, ("", ""))
    title_text = f"<b>{node_name}</b>  <span style='color:#888'>{role}  [{rows}×{cols}]</span>"
    if formula:
        title_text += f"   <span style='color:#aaa'>{formula}</span>"

    row_labels = [tokens[r] if r < len(tokens) else str(r) for r in range(rows)]
    if node_name in ("LastLogits", "Probs"):
        row_labels = ["last"]
    col_labels = [f"v{c}" for c in range(cols)]
    hover = [[f"{display[r, c]:.4f}" for c in range(cols)] for r in range(rows)]

    fig = go.Figure(go.Heatmap(
        z=display,
        colorscale="Blues" if node_name == "Probs" else "Viridis",
        showscale=True,
        hovertemplate="%{text}<extra></extra>",
        text=hover,
        x=col_labels,
        y=row_labels,
    ))
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=12)),
        height=270,
        margin=dict(l=50, r=10, t=48, b=30),
        paper_bgcolor="white",
        plot_bgcolor="white",
        yaxis=dict(autorange="reversed"),
    )
    return fig


# ---------------------------------------------------------------------------
# Multi-head attention flow
# ---------------------------------------------------------------------------

MULTI_HEAD_LAYOUT = {
    "X":      dict(cx=0.030, cy=0.50, w=0.045, h=0.105, nt="input"),
    "Wq1":    dict(cx=0.125, cy=0.900, w=0.040, h=0.060, nt="param", param=True),
    "Wk1":    dict(cx=0.125, cy=0.770, w=0.040, h=0.060, nt="param", param=True),
    "Wv1":    dict(cx=0.125, cy=0.640, w=0.040, h=0.060, nt="param", param=True),
    "Q1":     dict(cx=0.220, cy=0.900, w=0.042, h=0.060, nt="activation"),
    "K1":     dict(cx=0.220, cy=0.770, w=0.042, h=0.060, nt="activation"),
    "V1":     dict(cx=0.220, cy=0.640, w=0.042, h=0.060, nt="activation"),
    "A1":     dict(cx=0.385, cy=0.770, w=0.052, h=0.095, nt="attn"),
    "C1":     dict(cx=0.530, cy=0.770, w=0.052, h=0.095, nt="output"),
    "Wq2":    dict(cx=0.125, cy=0.360, w=0.040, h=0.060, nt="param", param=True),
    "Wk2":    dict(cx=0.125, cy=0.230, w=0.040, h=0.060, nt="param", param=True),
    "Wv2":    dict(cx=0.125, cy=0.100, w=0.040, h=0.060, nt="param", param=True),
    "Q2":     dict(cx=0.220, cy=0.360, w=0.042, h=0.060, nt="activation"),
    "K2":     dict(cx=0.220, cy=0.230, w=0.042, h=0.060, nt="activation"),
    "V2":     dict(cx=0.220, cy=0.100, w=0.042, h=0.060, nt="activation"),
    "A2":     dict(cx=0.385, cy=0.230, w=0.052, h=0.095, nt="attn"),
    "C2":     dict(cx=0.530, cy=0.230, w=0.052, h=0.095, nt="output"),
    "Concat": dict(cx=0.675, cy=0.50, w=0.104, h=0.095, nt="concat"),
    "Wo":     dict(cx=0.800, cy=0.50, w=0.056, h=0.085, nt="param", param=True),
    "MHAOut": dict(cx=0.940, cy=0.50, w=0.064, h=0.115, nt="output"),
}

MULTI_HEAD_ROLES = {
    "X":      ("Input", "shared by heads"),
    "Wq1":    ("Query Weight", "head 1"),
    "Wk1":    ("Key Weight", "head 1"),
    "Wv1":    ("Value Weight", "head 1"),
    "Q1":     ("Head 1 Query", "X @ Wq1"),
    "K1":     ("Head 1 Key", "X @ Wk1"),
    "V1":     ("Head 1 Value", "X @ Wv1"),
    "A1":     ("Head 1 Weights", "softmax(Q1K1ᵀ)"),
    "C1":     ("Head 1 Context", "A1 @ V1"),
    "Wq2":    ("Query Weight", "head 2"),
    "Wk2":    ("Key Weight", "head 2"),
    "Wv2":    ("Value Weight", "head 2"),
    "Q2":     ("Head 2 Query", "X @ Wq2"),
    "K2":     ("Head 2 Key", "X @ Wk2"),
    "V2":     ("Head 2 Value", "X @ Wv2"),
    "A2":     ("Head 2 Weights", "softmax(Q2K2ᵀ)"),
    "C2":     ("Head 2 Context", "A2 @ V2"),
    "Concat": ("Concatenate", "[C1 | C2]"),
    "Wo":     ("Output Weight", "learned param"),
    "MHAOut": ("MHA Output", "Concat @ Wo"),
}

_MULTI_HEAD_COLOR = {
    "input": "#7d3c98",
    "param": "#b7770d",
    "activation": "#c0392b",
    "attn": "#117a65",
    "concat": "#566573",
    "output": "#a04000",
}

MULTI_HEAD_EDGES = [
    ("X", "Wq1", "X->Wq1", ""),
    ("Wq1", "Q1", "Wq1->Q1", "X · Wq1"),
    ("X", "Wk1", "X->Wk1", ""),
    ("Wk1", "K1", "Wk1->K1", "X · Wk1"),
    ("X", "Wv1", "X->Wv1", ""),
    ("Wv1", "V1", "Wv1->V1", "X · Wv1"),
    ("X", "Wq2", "X->Wq2", ""),
    ("Wq2", "Q2", "Wq2->Q2", "X · Wq2"),
    ("X", "Wk2", "X->Wk2", ""),
    ("Wk2", "K2", "Wk2->K2", "X · Wk2"),
    ("X", "Wv2", "X->Wv2", ""),
    ("Wv2", "V2", "Wv2->V2", "X · Wv2"),
    ("Q1", "A1", "Q1->A1", "Q1 · K1ᵀ"),
    ("K1", "A1", "K1->A1", ""),
    ("Q2", "A2", "Q2->A2", "Q2 · K2ᵀ"),
    ("K2", "A2", "K2->A2", ""),
    ("A1", "C1", "A1->C1", "A1 · V1"),
    ("V1", "C1", "V1->C1", ""),
    ("A2", "C2", "A2->C2", "A2 · V2"),
    ("V2", "C2", "V2->C2", ""),
    ("C1", "Concat", "C1->Concat", "join columns"),
    ("C2", "Concat", "C2->Concat", ""),
    ("Concat", "Wo", "Concat->Wo", ""),
    ("Wo", "MHAOut", "Wo->MHAOut", "Concat · Wo"),
]


def _mh_bounds(name):
    n = MULTI_HEAD_LAYOUT[name]
    hw, hh = n["w"] / 2, n["h"] / 2
    return n["cx"] - hw, n["cy"] - hh, n["cx"] + hw, n["cy"] + hh


def _mh_port(name, side, y_frac=0.5):
    x0, y0, x1, y1 = _mh_bounds(name)
    py = y0 + y_frac * (y1 - y0)
    return (x1, py) if side == "right" else (x0, py)


def _mh_shape_label(name, mat):
    rows, cols = mat.shape
    return f"{rows}×{cols}"


def multi_head_diagram(matrices, active_nodes, active_edges, tokens,
                       fig_width=1500, fig_height=860):
    active_node_set = set(active_nodes)
    active_edge_set = set(active_edges)
    shapes = []
    annotations = []
    traces = []

    for name, layout in MULTI_HEAD_LAYOUT.items():
        is_active = name in active_node_set
        color = _MULTI_HEAD_COLOR[layout["nt"]]
        has_params = layout.get("param", False)
        x0, y0, x1, y1 = _mh_bounds(name)

        shapes.append(dict(
            type="rect", xref="paper", yref="paper",
            x0=x0, y0=y0, x1=x1, y1=y1,
            fillcolor=_rgba(color, 0.12 if is_active else 0.035),
            line=dict(
                color=_rgba(color, 1.0 if is_active else 0.18),
                width=2.3 if is_active else 1.0,
                dash="dash" if has_params else "solid",
            ),
            layer="below",
        ))

        role, formula = MULTI_HEAD_ROLES[name]
        compact_node = layout["nt"] in ("param", "activation")
        title_size = 13 if compact_node else 17
        role_size = 9 if compact_node else 11
        role_line = "" if compact_node else (
            f"<br><span style='font-size:{role_size}px;color:{'#333' if is_active else '#ccc'}'>{role}</span>"
        )
        annotations.append(dict(
            xref="paper", yref="paper", x=layout["cx"], y=y1 + 0.016,
            yanchor="bottom", showarrow=False, align="center",
            text=(f"<span style='font-size:{title_size}px;font-weight:800;color:{color if is_active else _rgba(color, 0.30)}'>{name}</span>"
                  f"{role_line}"),
        ))

        mat = matrices.get(name)
        if mat is not None:
            formula_text = "" if layout["nt"] in ("param", "activation") else (
                f"<br><span style='font-size:8px;color:{'#666' if is_active else '#ddd'}'>{formula}</span>"
            )
            annotations.append(dict(
                xref="paper", yref="paper", x=layout["cx"], y=y0 - 0.018,
                yanchor="top", showarrow=False, align="center",
                text=(f"<span style='font-size:9px;color:{'#333' if is_active else '#bbb'}'>[{_mh_shape_label(name, mat)}]</span>"
                      f"{formula_text}"),
            ))

    for src, dst, edge_key, label in MULTI_HEAD_EDGES:
        is_active = edge_key in active_edge_set
        alpha = 0.86 if is_active else 0.10
        width = 2.0 if is_active else 0.7
        sx, sy = _mh_port(src, "right")
        dx, dy = _mh_port(dst, "left")
        route = "short" if src.startswith("W") or dst.startswith("W") or src in ("Concat", "Wo") else "scurve"
        path = _bezier_path(sx, sy, dx, dy, route)
        shapes.append(dict(
            type="path", xref="paper", yref="paper", path=path,
            fillcolor="rgba(0,0,0,0)",
            line=dict(color=_rgba("#1c2833", alpha), width=width),
            layer="above",
        ))
        annotations.append(dict(
            xref="paper", yref="paper", x=dx, y=dy,
            ax=-16, ay=0, axref="pixel", ayref="pixel",
            showarrow=True, arrowhead=3, arrowsize=0.85, arrowwidth=width,
            arrowcolor=_rgba("#1c2833", alpha), text="",
        ))
        if label and is_active:
            label_y = max(sy, dy) + 0.024
            if src.startswith("W"):
                label_y = sy + 0.018
            elif edge_key in ("Q1->A1", "Q2->A2"):
                label_y = max(sy, dy) + 0.055
            elif edge_key in ("A1->C1", "A2->C2"):
                label_y = sy + 0.035
            annotations.append(dict(
                xref="paper", yref="paper", x=(sx + dx) / 2, y=label_y,
                text=f"<b>{label}</b>", showarrow=False,
                bgcolor="rgba(255,255,255,0.90)", borderpad=2,
                font=dict(size=8 if src.startswith("W") else 9, color="#1c2833"),
            ))

    annotations.extend([
        dict(xref="paper", yref="paper", x=0.385, y=0.990,
             text="<b>Head 1</b>", showarrow=False, font=dict(size=16, color="#b03a2e")),
        dict(xref="paper", yref="paper", x=0.385, y=0.010,
             text="<b>Head 2</b>", showarrow=False, font=dict(size=16, color="#2874a6")),
    ])

    node_list = list(MULTI_HEAD_LAYOUT.keys())
    for name, layout in MULTI_HEAD_LAYOUT.items():
        mat = matrices.get(name)
        if mat is None:
            continue
        is_active = name in active_node_set
        display = mat.copy().astype(float)
        rows, cols = display.shape
        use_div = name.startswith("A")
        is_param_node = layout["nt"] == "param"
        hover = [[f"<b>{name}[{r},{c}]</b><br>{display[r, c]:.4f}"
                  for c in range(cols)] for r in range(rows)]

        idx = node_list.index(name)
        xax = "x" if idx == 0 else f"x{idx+1}"
        yax = "y" if idx == 0 else f"y{idx+1}"
        xak = "xaxis" if idx == 0 else f"xaxis{idx+1}"
        yak = "yaxis" if idx == 0 else f"yaxis{idx+1}"
        x0, y0, x1, y1 = _mh_bounds(name)
        row_labels = [tokens[r] if r < len(tokens) else str(r) for r in range(rows)]

        trace = go.Heatmap(
            z=display,
            colorscale="Blues" if use_div else ("YlOrBr" if is_param_node else "Viridis"),
            showscale=False,
            opacity=0.92 if is_active else 0.18,
            hoverinfo="text",
            hovertemplate="%{text}<extra></extra>",
            text=hover,
            y=row_labels if name == "X" else None,
            xaxis=xax,
            yaxis=yax,
        )
        traces.append((trace, xak, yak, x0, y0, x1, y1, name))

    layout_args = dict(
        paper_bgcolor="white", plot_bgcolor="white",
        width=fig_width, height=fig_height,
        margin=dict(l=8, r=8, t=58, b=48),
        showlegend=False,
        shapes=shapes,
        annotations=annotations,
    )

    for _, xak, yak, x0, y0, x1, y1, name in traces:
        layout_args[xak] = dict(
            domain=[x0, x1], showticklabels=False,
            showgrid=False, zeroline=False, fixedrange=True,
        )
        layout_args[yak] = dict(
            domain=[y0, y1], showticklabels=(name == "X"),
            tickfont=dict(size=8, color="#444"),
            side="right",
            showgrid=False, zeroline=False, fixedrange=True,
            autorange="reversed",
        )

    return go.Figure(data=[t[0] for t in traces], layout=go.Layout(**layout_args))


def multi_head_detail_panel(node_name, matrices, tokens):
    mat = matrices.get(node_name)
    if mat is None:
        role, formula = MULTI_HEAD_ROLES.get(node_name, ("", ""))
        fig = go.Figure()
        fig.update_layout(
            title=dict(text=f"<b>{node_name}</b>  <span style='color:#888'>{role}</span>",
                       font=dict(size=12)),
            height=270,
            margin=dict(l=20, r=20, t=48, b=20),
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[dict(
                x=0.5, y=0.52, xref="paper", yref="paper",
                text=formula, showarrow=False, font=dict(size=15, color="#333"),
            )],
        )
        return fig

    display = mat.copy().astype(float)
    rows, cols = display.shape
    role, formula = MULTI_HEAD_ROLES.get(node_name, ("", ""))
    title_text = f"<b>{node_name}</b>  <span style='color:#888'>{role}  [{rows}×{cols}]</span>"
    if formula:
        title_text += f"   <span style='color:#aaa'>{formula}</span>"

    row_labels = [tokens[r] if r < len(tokens) else str(r) for r in range(rows)]
    col_labels = [f"d{c}" for c in range(cols)]
    hover = [[f"{display[r, c]:.4f}" for c in range(cols)] for r in range(rows)]

    fig = go.Figure(go.Heatmap(
        z=display,
        colorscale="Blues" if node_name.startswith("A") else "Viridis",
        showscale=True,
        hovertemplate="%{text}<extra></extra>",
        text=hover,
        x=col_labels,
        y=row_labels,
    ))
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=12)),
        height=270,
        margin=dict(l=50, r=10, t=48, b=30),
        paper_bgcolor="white",
        plot_bgcolor="white",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def multi_head_kv_cache_diagram(matrices, active_nodes, active_edges, tokens, decode_row_view=False):
    fig = multi_head_diagram(matrices, active_nodes, active_edges, tokens)
    shapes = list(fig.layout.shapes)
    annotations = list(fig.layout.annotations)
    seq_len = matrices["X"].shape[0]
    new_row = seq_len - 1
    cached_color = "#0057b8"
    new_color = "#d35400"

    if decode_row_view:
        _blank_old_rows(shapes, _mh_bounds, ("X", "Q1", "Q2", "A1", "A2", "C1", "C2", "Concat", "MHAOut"), seq_len)

    for node_name in ("X", "Q1", "K1", "V1", "Q2", "K2", "V2", "A1", "A2", "C1", "C2", "Concat", "MHAOut"):
        x0, y0, x1, y1 = _mh_bounds(node_name)
        band_y0, band_y1 = _row_band((x0, y0, x1, y1), seq_len, new_row)
        shapes.append(dict(
            type="rect", xref="paper", yref="paper",
            x0=x0, y0=band_y0, x1=x1, y1=band_y1,
            fillcolor=_rgba(new_color, 0.34),
            line=dict(color=_rgba(new_color, 0.95), width=2.2),
            layer="above",
        ))

    for row in range(seq_len - 1):
        for node_name in ("K1", "V1", "K2", "V2"):
            x0, y0, x1, y1 = _mh_bounds(node_name)
            band_y0, band_y1 = _row_band((x0, y0, x1, y1), seq_len, row)
            shapes.append(dict(
                type="rect", xref="paper", yref="paper",
                x0=x0, y0=band_y0, x1=x1, y1=band_y1,
                fillcolor=_rgba(cached_color, 0.34),
                line=dict(color=_rgba(cached_color, 1.0), width=2.4),
                layer="above",
            ))

    annotations.extend([
        dict(xref="paper", yref="paper", x=0.560, y=1.105,
             text="<span style='color:#d35400'><b>orange</b></span> = newest token rows",
             showarrow=False, xanchor="left", bgcolor="rgba(255,255,255,0.92)",
             borderpad=3, font=dict(size=11, color="#333")),
        dict(xref="paper", yref="paper", x=0.560, y=1.065,
             text="<span style='color:#0057b8'><b>blue</b></span> = separate cached K/V rows per head",
             showarrow=False, xanchor="left", bgcolor="rgba(255,255,255,0.92)",
             borderpad=3, font=dict(size=11, color="#333")),
        dict(xref="paper", yref="paper", x=0.560, y=1.025,
             text="Q rows are computed for the newest token only; Q is not cached",
             showarrow=False, xanchor="left", bgcolor="rgba(255,255,255,0.92)",
             borderpad=3, font=dict(size=11, color="#333")),
        dict(xref="paper", yref="paper", x=0.560, y=0.985,
             text="Only newest C1/C2 rows form the newest Concat row for logits",
             showarrow=False, xanchor="left", bgcolor="rgba(255,255,255,0.92)",
             borderpad=3, font=dict(size=11, color="#333")),
    ])
    if decode_row_view:
        annotations.append(dict(xref="paper", yref="paper", x=0.560, y=0.945,
             text="Decode row view: old non-cache rows are hidden",
             showarrow=False, xanchor="left", bgcolor="rgba(255,255,255,0.92)",
             borderpad=3, font=dict(size=11, color="#333")))
    fig.update_layout(margin=dict(l=8, r=8, t=100, b=48))
    fig.update_layout(shapes=shapes, annotations=annotations)
    return fig


# ---------------------------------------------------------------------------
# Multi-query attention flow
# ---------------------------------------------------------------------------

MQA_LAYOUT = {
    "X":      dict(cx=0.035, cy=0.50, w=0.050, h=0.120, nt="input"),
    "Wq1":    dict(cx=0.145, cy=0.82, w=0.044, h=0.070, nt="param", param=True),
    "Wq2":    dict(cx=0.145, cy=0.24, w=0.044, h=0.070, nt="param", param=True),
    "Q1":     dict(cx=0.240, cy=0.82, w=0.048, h=0.078, nt="activation"),
    "Q2":     dict(cx=0.240, cy=0.24, w=0.048, h=0.078, nt="activation"),
    "Wk":     dict(cx=0.145, cy=0.58, w=0.044, h=0.070, nt="param", param=True),
    "Wv":     dict(cx=0.145, cy=0.42, w=0.044, h=0.070, nt="param", param=True),
    "K":      dict(cx=0.240, cy=0.58, w=0.048, h=0.078, nt="activation"),
    "V":      dict(cx=0.240, cy=0.42, w=0.048, h=0.078, nt="activation"),
    "A1":     dict(cx=0.390, cy=0.74, w=0.060, h=0.115, nt="attn"),
    "A2":     dict(cx=0.390, cy=0.26, w=0.060, h=0.115, nt="attn"),
    "C1":     dict(cx=0.535, cy=0.74, w=0.060, h=0.115, nt="output"),
    "C2":     dict(cx=0.535, cy=0.26, w=0.060, h=0.115, nt="output"),
    "Concat": dict(cx=0.690, cy=0.50, w=0.120, h=0.115, nt="concat"),
    "Wo":     dict(cx=0.810, cy=0.50, w=0.060, h=0.095, nt="param", param=True),
    "MQAOut": dict(cx=0.940, cy=0.50, w=0.070, h=0.125, nt="output"),
}

MQA_ROLES = {
    "X":      ("Input", "shared input"),
    "Wq1":    ("Query Weight", "head 1"),
    "Wq2":    ("Query Weight", "head 2"),
    "Q1":     ("Head 1 Query", "X @ Wq1"),
    "Q2":     ("Head 2 Query", "X @ Wq2"),
    "Wk":     ("Shared Key Weight", "one K path"),
    "Wv":     ("Shared Value Weight", "one V path"),
    "K":      ("Shared Keys", "X @ Wk"),
    "V":      ("Shared Values", "X @ Wv"),
    "A1":     ("Head 1 Weights", "softmax(Q1Kᵀ)"),
    "A2":     ("Head 2 Weights", "softmax(Q2Kᵀ)"),
    "C1":     ("Head 1 Context", "A1 @ V"),
    "C2":     ("Head 2 Context", "A2 @ V"),
    "Concat": ("Concatenate", "[C1 | C2]"),
    "Wo":     ("Output Weight", "learned param"),
    "MQAOut": ("MQA Output", "Concat @ Wo"),
}

_MQA_COLOR = {
    "input": "#7d3c98",
    "param": "#b7770d",
    "activation": "#c0392b",
    "attn": "#117a65",
    "concat": "#566573",
    "output": "#a04000",
}

MQA_EDGES = [
    ("X", "Wq1", "X->Wq1", ""),
    ("Wq1", "Q1", "Wq1->Q1", "X · Wq1"),
    ("X", "Wq2", "X->Wq2", ""),
    ("Wq2", "Q2", "Wq2->Q2", "X · Wq2"),
    ("X", "Wk", "X->Wk", ""),
    ("Wk", "K", "Wk->K", "X · Wk"),
    ("X", "Wv", "X->Wv", ""),
    ("Wv", "V", "Wv->V", "X · Wv"),
    ("Q1", "A1", "Q1->A1", "Q1 · Kᵀ"),
    ("K", "A1", "K->A1", ""),
    ("Q2", "A2", "Q2->A2", "Q2 · Kᵀ"),
    ("K", "A2", "K->A2", ""),
    ("A1", "C1", "A1->C1", "A1 · V"),
    ("V", "C1", "V->C1", ""),
    ("A2", "C2", "A2->C2", "A2 · V"),
    ("V", "C2", "V->C2", ""),
    ("C1", "Concat", "C1->Concat", "join columns"),
    ("C2", "Concat", "C2->Concat", ""),
    ("Concat", "Wo", "Concat->Wo", ""),
    ("Wo", "MQAOut", "Wo->MQAOut", "Concat · Wo"),
]


def _mqa_bounds(name):
    n = MQA_LAYOUT[name]
    hw, hh = n["w"] / 2, n["h"] / 2
    return n["cx"] - hw, n["cy"] - hh, n["cx"] + hw, n["cy"] + hh


def _mqa_port(name, side, y_frac=0.5):
    x0, y0, x1, y1 = _mqa_bounds(name)
    py = y0 + y_frac * (y1 - y0)
    return (x1, py) if side == "right" else (x0, py)


def _mqa_shape_label(name, mat):
    rows, cols = mat.shape
    return f"{rows}×{cols}"


def mqa_diagram(matrices, active_nodes, active_edges, tokens,
                fig_width=1500, fig_height=760):
    active_node_set = set(active_nodes)
    active_edge_set = set(active_edges)
    shapes = []
    annotations = []
    traces = []

    for name, layout in MQA_LAYOUT.items():
        is_active = name in active_node_set
        color = _MQA_COLOR[layout["nt"]]
        has_params = layout.get("param", False)
        x0, y0, x1, y1 = _mqa_bounds(name)
        shapes.append(dict(
            type="rect", xref="paper", yref="paper",
            x0=x0, y0=y0, x1=x1, y1=y1,
            fillcolor=_rgba(color, 0.12 if is_active else 0.035),
            line=dict(color=_rgba(color, 1.0 if is_active else 0.18),
                      width=2.3 if is_active else 1.0,
                      dash="dash" if has_params else "solid"),
            layer="below",
        ))

        role, formula = MQA_ROLES[name]
        compact_node = layout["nt"] in ("param", "activation")
        title_size = 14 if compact_node else 17
        role_line = "" if compact_node else (
            f"<br><span style='font-size:11px;color:{'#333' if is_active else '#ccc'}'>{role}</span>"
        )
        annotations.append(dict(
            xref="paper", yref="paper", x=layout["cx"], y=y1 + 0.016,
            yanchor="bottom", showarrow=False, align="center",
            text=(f"<span style='font-size:{title_size}px;font-weight:800;color:{color if is_active else _rgba(color, 0.30)}'>{name}</span>"
                  f"{role_line}"),
        ))

        mat = matrices.get(name)
        if mat is not None:
            formula_text = "" if layout["nt"] in ("param", "activation") else (
                f"<br><span style='font-size:8px;color:{'#666' if is_active else '#ddd'}'>{formula}</span>"
            )
            annotations.append(dict(
                xref="paper", yref="paper", x=layout["cx"], y=y0 - 0.018,
                yanchor="top", showarrow=False, align="center",
                text=(f"<span style='font-size:9px;color:{'#333' if is_active else '#bbb'}'>[{_mqa_shape_label(name, mat)}]</span>"
                      f"{formula_text}"),
            ))

    for src, dst, edge_key, label in MQA_EDGES:
        is_active = edge_key in active_edge_set
        alpha = 0.86 if is_active else 0.10
        width = 2.0 if is_active else 0.7
        sx, sy = _mqa_port(src, "right")
        dx, dy = _mqa_port(dst, "left")
        route = "short" if src.startswith("W") or dst.startswith("W") or src in ("Concat", "Wo") else "scurve"
        path = _bezier_path(sx, sy, dx, dy, route)
        shapes.append(dict(
            type="path", xref="paper", yref="paper", path=path,
            fillcolor="rgba(0,0,0,0)",
            line=dict(color=_rgba("#1c2833", alpha), width=width),
            layer="above",
        ))
        annotations.append(dict(
            xref="paper", yref="paper", x=dx, y=dy,
            ax=-16, ay=0, axref="pixel", ayref="pixel",
            showarrow=True, arrowhead=3, arrowsize=0.85, arrowwidth=width,
            arrowcolor=_rgba("#1c2833", alpha), text="",
        ))
        if label and is_active:
            label_y = max(sy, dy) + 0.030
            if src.startswith("W"):
                label_y = sy + 0.018
            elif edge_key in ("Q1->A1", "Q2->A2"):
                label_y = max(sy, dy) + 0.055
            annotations.append(dict(
                xref="paper", yref="paper", x=(sx + dx) / 2, y=label_y,
                text=f"<b>{label}</b>", showarrow=False,
                bgcolor="rgba(255,255,255,0.90)", borderpad=2,
                font=dict(size=8 if src.startswith("W") else 9, color="#1c2833"),
            ))

    annotations.extend([
        dict(xref="paper", yref="paper", x=0.390, y=0.950,
             text="<b>Query Head 1</b>", showarrow=False, font=dict(size=15, color="#b03a2e")),
        dict(xref="paper", yref="paper", x=0.390, y=0.050,
             text="<b>Query Head 2</b>", showarrow=False, font=dict(size=15, color="#b03a2e")),
        dict(xref="paper", yref="paper", x=0.270, y=0.500,
             text="<b>shared K/V</b>", showarrow=False, font=dict(size=14, color="#117a65")),
    ])

    node_list = list(MQA_LAYOUT.keys())
    for name, layout in MQA_LAYOUT.items():
        mat = matrices.get(name)
        if mat is None:
            continue
        is_active = name in active_node_set
        display = mat.copy().astype(float)
        rows, cols = display.shape
        is_param_node = layout["nt"] == "param"
        hover = [[f"<b>{name}[{r},{c}]</b><br>{display[r, c]:.4f}"
                  for c in range(cols)] for r in range(rows)]
        idx = node_list.index(name)
        xax = "x" if idx == 0 else f"x{idx+1}"
        yax = "y" if idx == 0 else f"y{idx+1}"
        xak = "xaxis" if idx == 0 else f"xaxis{idx+1}"
        yak = "yaxis" if idx == 0 else f"yaxis{idx+1}"
        x0, y0, x1, y1 = _mqa_bounds(name)
        row_labels = [tokens[r] if r < len(tokens) else str(r) for r in range(rows)]
        trace = go.Heatmap(
            z=display,
            colorscale="Blues" if name.startswith("A") else ("YlOrBr" if is_param_node else "Viridis"),
            showscale=False,
            opacity=0.92 if is_active else 0.18,
            hoverinfo="text",
            hovertemplate="%{text}<extra></extra>",
            text=hover,
            y=row_labels if name == "X" else None,
            xaxis=xax,
            yaxis=yax,
        )
        traces.append((trace, xak, yak, x0, y0, x1, y1, name))

    layout_args = dict(
        paper_bgcolor="white", plot_bgcolor="white",
        width=fig_width, height=fig_height,
        margin=dict(l=8, r=8, t=58, b=48),
        showlegend=False,
        shapes=shapes,
        annotations=annotations,
    )
    for _, xak, yak, x0, y0, x1, y1, name in traces:
        layout_args[xak] = dict(domain=[x0, x1], showticklabels=False,
                                showgrid=False, zeroline=False, fixedrange=True)
        layout_args[yak] = dict(domain=[y0, y1], showticklabels=(name == "X"),
                                tickfont=dict(size=8, color="#444"), side="right",
                                showgrid=False, zeroline=False, fixedrange=True,
                                autorange="reversed")

    return go.Figure(data=[t[0] for t in traces], layout=go.Layout(**layout_args))


def mqa_kv_cache_diagram(matrices, active_nodes, active_edges, tokens, decode_row_view=False):
    fig = mqa_diagram(matrices, active_nodes, active_edges, tokens)
    shapes = list(fig.layout.shapes)
    annotations = list(fig.layout.annotations)
    seq_len = matrices["X"].shape[0]
    new_row = seq_len - 1
    cached_color = "#0057b8"
    new_color = "#d35400"

    if decode_row_view:
        _blank_old_rows(shapes, _mqa_bounds, ("X", "Q1", "Q2", "A1", "A2", "C1", "C2", "Concat", "MQAOut"), seq_len)

    for node_name in ("X", "Q1", "Q2", "K", "V", "A1", "A2", "C1", "C2", "Concat", "MQAOut"):
        x0, y0, x1, y1 = _mqa_bounds(node_name)
        band_y0, band_y1 = _row_band((x0, y0, x1, y1), seq_len, new_row)
        shapes.append(dict(
            type="rect", xref="paper", yref="paper",
            x0=x0, y0=band_y0, x1=x1, y1=band_y1,
            fillcolor=_rgba(new_color, 0.34),
            line=dict(color=_rgba(new_color, 0.95), width=2.2),
            layer="above",
        ))

    for row in range(seq_len - 1):
        for node_name in ("K", "V"):
            x0, y0, x1, y1 = _mqa_bounds(node_name)
            band_y0, band_y1 = _row_band((x0, y0, x1, y1), seq_len, row)
            shapes.append(dict(
                type="rect", xref="paper", yref="paper",
                x0=x0, y0=band_y0, x1=x1, y1=band_y1,
                fillcolor=_rgba(cached_color, 0.34),
                line=dict(color=_rgba(cached_color, 1.0), width=2.4),
                layer="above",
            ))

    annotations.extend([
        dict(xref="paper", yref="paper", x=0.610, y=1.185,
             text="<span style='color:#d35400'><b>orange</b></span> = newest token rows",
             showarrow=False, xanchor="left", bgcolor="rgba(255,255,255,0.92)",
             borderpad=3, font=dict(size=11, color="#333")),
        dict(xref="paper", yref="paper", x=0.610, y=1.145,
             text="<span style='color:#0057b8'><b>blue</b></span> = one shared cached K/V reused by all Q heads",
             showarrow=False, xanchor="left", bgcolor="rgba(255,255,255,0.92)",
             borderpad=3, font=dict(size=11, color="#333")),
        dict(xref="paper", yref="paper", x=0.610, y=1.105,
             text="Q rows are computed for the newest token only; Q is not cached",
             showarrow=False, xanchor="left", bgcolor="rgba(255,255,255,0.92)",
             borderpad=3, font=dict(size=11, color="#333")),
        dict(xref="paper", yref="paper", x=0.610, y=1.065,
             text="Only newest C1/C2 rows form the newest Concat row for logits",
             showarrow=False, xanchor="left", bgcolor="rgba(255,255,255,0.92)",
             borderpad=3, font=dict(size=11, color="#333")),
    ])
    if decode_row_view:
        annotations.append(dict(xref="paper", yref="paper", x=0.610, y=1.025,
             text="Decode row view: old non-cache rows are hidden",
             showarrow=False, xanchor="left", bgcolor="rgba(255,255,255,0.92)",
             borderpad=3, font=dict(size=11, color="#333")))
    fig.update_layout(margin=dict(l=8, r=8, t=150, b=48))
    fig.update_layout(shapes=shapes, annotations=annotations)
    return fig


def mqa_detail_panel(node_name, matrices, tokens):
    mat = matrices.get(node_name)
    if mat is None:
        return go.Figure()
    display = mat.copy().astype(float)
    rows, cols = display.shape
    role, formula = MQA_ROLES.get(node_name, ("", ""))
    title_text = f"<b>{node_name}</b>  <span style='color:#888'>{role}  [{rows}×{cols}]</span>"
    if formula:
        title_text += f"   <span style='color:#aaa'>{formula}</span>"
    row_labels = [tokens[r] if r < len(tokens) else str(r) for r in range(rows)]
    col_labels = [f"d{c}" for c in range(cols)]
    hover = [[f"{display[r, c]:.4f}" for c in range(cols)] for r in range(rows)]
    fig = go.Figure(go.Heatmap(
        z=display,
        colorscale="Blues" if node_name.startswith("A") else "Viridis",
        showscale=True,
        hovertemplate="%{text}<extra></extra>",
        text=hover,
        x=col_labels,
        y=row_labels,
    ))
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=12)),
        height=270,
        margin=dict(l=50, r=10, t=48, b=30),
        paper_bgcolor="white",
        plot_bgcolor="white",
        yaxis=dict(autorange="reversed"),
    )
    return fig


# ---------------------------------------------------------------------------
# Grouped-query attention KV cache flow
# ---------------------------------------------------------------------------

GQA_LAYOUT = {
    "X":      dict(cx=0.030, cy=0.50, w=0.044, h=0.085, nt="input"),
    "Wq1":    dict(cx=0.110, cy=0.805, w=0.036, h=0.048, nt="param", param=True),
    "Q1":     dict(cx=0.185, cy=0.805, w=0.038, h=0.052, nt="activation"),
    "Wq2":    dict(cx=0.110, cy=0.555, w=0.036, h=0.048, nt="param", param=True),
    "Q2":     dict(cx=0.185, cy=0.555, w=0.038, h=0.052, nt="activation"),
    "WkG1":   dict(cx=0.270, cy=0.735, w=0.040, h=0.052, nt="param", param=True),
    "KG1":    dict(cx=0.350, cy=0.735, w=0.044, h=0.064, nt="activation"),
    "WvG1":   dict(cx=0.270, cy=0.625, w=0.040, h=0.052, nt="param", param=True),
    "VG1":    dict(cx=0.350, cy=0.625, w=0.044, h=0.064, nt="activation"),
    "A1":     dict(cx=0.485, cy=0.805, w=0.046, h=0.080, nt="attn"),
    "C1":     dict(cx=0.595, cy=0.805, w=0.046, h=0.080, nt="output"),
    "A2":     dict(cx=0.485, cy=0.555, w=0.046, h=0.080, nt="attn"),
    "C2":     dict(cx=0.595, cy=0.555, w=0.046, h=0.080, nt="output"),

    "Wq3":    dict(cx=0.110, cy=0.405, w=0.036, h=0.048, nt="param", param=True),
    "Q3":     dict(cx=0.185, cy=0.405, w=0.038, h=0.052, nt="activation"),
    "Wq4":    dict(cx=0.110, cy=0.155, w=0.036, h=0.048, nt="param", param=True),
    "Q4":     dict(cx=0.185, cy=0.155, w=0.038, h=0.052, nt="activation"),
    "WkG2":   dict(cx=0.270, cy=0.335, w=0.040, h=0.052, nt="param", param=True),
    "KG2":    dict(cx=0.350, cy=0.335, w=0.044, h=0.064, nt="activation"),
    "WvG2":   dict(cx=0.270, cy=0.225, w=0.040, h=0.052, nt="param", param=True),
    "VG2":    dict(cx=0.350, cy=0.225, w=0.044, h=0.064, nt="activation"),
    "A3":     dict(cx=0.485, cy=0.405, w=0.046, h=0.080, nt="attn"),
    "C3":     dict(cx=0.595, cy=0.405, w=0.046, h=0.080, nt="output"),
    "A4":     dict(cx=0.485, cy=0.155, w=0.046, h=0.080, nt="attn"),
    "C4":     dict(cx=0.595, cy=0.155, w=0.046, h=0.080, nt="output"),

    "Concat": dict(cx=0.735, cy=0.500, w=0.150, h=0.090, nt="concat"),
    "Wo":     dict(cx=0.865, cy=0.500, w=0.056, h=0.082, nt="param", param=True),
    "GQAOut": dict(cx=0.955, cy=0.500, w=0.060, h=0.100, nt="output"),
}

GQA_ROLES = {
    "X": ("Input", "shared by all heads"),
    "Wq1": ("Query Weight", "head 1"), "Q1": ("Head 1 Query", "X @ Wq1"),
    "Wq2": ("Query Weight", "head 2"), "Q2": ("Head 2 Query", "X @ Wq2"),
    "Wq3": ("Query Weight", "head 3"), "Q3": ("Head 3 Query", "X @ Wq3"),
    "Wq4": ("Query Weight", "head 4"), "Q4": ("Head 4 Query", "X @ Wq4"),
    "WkG1": ("Group 1 Key Weight", "learned param"),
    "WvG1": ("Group 1 Value Weight", "learned param"),
    "KG1": ("Group 1 Keys", "shared by heads 1-2"),
    "VG1": ("Group 1 Values", "shared by heads 1-2"),
    "WkG2": ("Group 2 Key Weight", "learned param"),
    "WvG2": ("Group 2 Value Weight", "learned param"),
    "KG2": ("Group 2 Keys", "shared by heads 3-4"),
    "VG2": ("Group 2 Values", "shared by heads 3-4"),
    "A1": ("Head 1 Weights", "softmax(Q1KG1ᵀ)"),
    "A2": ("Head 2 Weights", "softmax(Q2KG1ᵀ)"),
    "A3": ("Head 3 Weights", "softmax(Q3KG2ᵀ)"),
    "A4": ("Head 4 Weights", "softmax(Q4KG2ᵀ)"),
    "C1": ("Head 1 Context", "A1 @ VG1"),
    "C2": ("Head 2 Context", "A2 @ VG1"),
    "C3": ("Head 3 Context", "A3 @ VG2"),
    "C4": ("Head 4 Context", "A4 @ VG2"),
    "Concat": ("Concatenate", "[C1 | C2 | C3 | C4]"),
    "Wo": ("Output Weight", "learned param"),
    "GQAOut": ("GQA Output", "Concat @ Wo"),
}

_GQA_COLOR = {
    "input": "#7d3c98",
    "param": "#b7770d",
    "activation": "#c0392b",
    "attn": "#117a65",
    "concat": "#566573",
    "output": "#a04000",
}

GQA_EDGES = [
    ("X", "Wq1", "X->Wq1", ""), ("Wq1", "Q1", "Wq1->Q1", "X · Wq1"),
    ("X", "Wq2", "X->Wq2", ""), ("Wq2", "Q2", "Wq2->Q2", "X · Wq2"),
    ("X", "Wq3", "X->Wq3", ""), ("Wq3", "Q3", "Wq3->Q3", "X · Wq3"),
    ("X", "Wq4", "X->Wq4", ""), ("Wq4", "Q4", "Wq4->Q4", "X · Wq4"),
    ("X", "WkG1", "X->WkG1", ""), ("WkG1", "KG1", "WkG1->KG1", "X · WkG1"),
    ("X", "WvG1", "X->WvG1", ""), ("WvG1", "VG1", "WvG1->VG1", "X · WvG1"),
    ("X", "WkG2", "X->WkG2", ""), ("WkG2", "KG2", "WkG2->KG2", "X · WkG2"),
    ("X", "WvG2", "X->WvG2", ""), ("WvG2", "VG2", "WvG2->VG2", "X · WvG2"),
    ("Q1", "A1", "Q1->A1", "Q1 · KG1ᵀ"), ("KG1", "A1", "KG1->A1", ""),
    ("Q2", "A2", "Q2->A2", "Q2 · KG1ᵀ"), ("KG1", "A2", "KG1->A2", ""),
    ("Q3", "A3", "Q3->A3", "Q3 · KG2ᵀ"), ("KG2", "A3", "KG2->A3", ""),
    ("Q4", "A4", "Q4->A4", "Q4 · KG2ᵀ"), ("KG2", "A4", "KG2->A4", ""),
    ("A1", "C1", "A1->C1", "A1 · VG1"), ("VG1", "C1", "VG1->C1", ""),
    ("A2", "C2", "A2->C2", "A2 · VG1"), ("VG1", "C2", "VG1->C2", ""),
    ("A3", "C3", "A3->C3", "A3 · VG2"), ("VG2", "C3", "VG2->C3", ""),
    ("A4", "C4", "A4->C4", "A4 · VG2"), ("VG2", "C4", "VG2->C4", ""),
    ("C1", "Concat", "C1->Concat", "join columns"),
    ("C2", "Concat", "C2->Concat", ""),
    ("C3", "Concat", "C3->Concat", ""),
    ("C4", "Concat", "C4->Concat", ""),
    ("Concat", "Wo", "Concat->Wo", ""),
    ("Wo", "GQAOut", "Wo->GQAOut", "Concat · Wo"),
]


def _gqa_bounds(name):
    n = GQA_LAYOUT[name]
    hw, hh = n["w"] / 2, n["h"] / 2
    return n["cx"] - hw, n["cy"] - hh, n["cx"] + hw, n["cy"] + hh


def _gqa_port(name, side, y_frac=0.5):
    x0, y0, x1, y1 = _gqa_bounds(name)
    py = y0 + y_frac * (y1 - y0)
    return (x1, py) if side == "right" else (x0, py)


def _gqa_shape_label(name, mat):
    rows, cols = mat.shape
    return f"{rows}×{cols}"


def _gqa_compact_label_side(name):
    below_nodes = {
        "Wq2", "Q2", "WvG1", "VG1", "A2", "C2",
        "Wq4", "Q4", "WvG2", "VG2", "A4", "C4",
    }
    return "below" if name in below_nodes else "above"


def gqa_kv_cache_diagram(matrices, active_nodes, active_edges, tokens,
                         decode_row_view=False,
                         fig_width=1600, fig_height=1280):
    active_node_set = set(active_nodes)
    active_edge_set = set(active_edges)
    shapes = []
    annotations = []
    traces = []
    seq_len = matrices["X"].shape[0]
    new_row = seq_len - 1
    cached_color = "#0057b8"
    new_color = "#d35400"

    for name, layout in GQA_LAYOUT.items():
        is_active = name in active_node_set
        color = _GQA_COLOR[layout["nt"]]
        has_params = layout.get("param", False)
        x0, y0, x1, y1 = _gqa_bounds(name)
        shapes.append(dict(
            type="rect", xref="paper", yref="paper",
            x0=x0, y0=y0, x1=x1, y1=y1,
            fillcolor=_rgba(color, 0.12 if is_active else 0.035),
            line=dict(color=_rgba(color, 1.0 if is_active else 0.18),
                      width=2.3 if is_active else 1.0,
                      dash="dash" if has_params else "solid"),
            layer="below",
        ))

        role, formula = GQA_ROLES[name]
        mat = matrices.get(name)
        compact_node = layout["nt"] in ("param", "activation")
        title_size = 12 if compact_node else 16

        if compact_node:
            label_side = _gqa_compact_label_side(name)
            label_y = y0 - 0.012 if label_side == "below" else y1 + 0.012
            yanchor = "top" if label_side == "below" else "bottom"
            dim_line = (
                f"<br><span style='font-size:8px;color:{'#333' if is_active else '#bbb'}'>[{_gqa_shape_label(name, mat)}]</span>"
                if mat is not None else ""
            )
            annotations.append(dict(
                xref="paper", yref="paper", x=layout["cx"], y=label_y,
                yanchor=yanchor, showarrow=False, align="center",
                text=(f"<span style='font-size:{title_size}px;font-weight:800;color:{color if is_active else _rgba(color, 0.30)}'>{name}</span>"
                      f"{dim_line}"),
            ))
            continue

        role_line = (
            f"<br><span style='font-size:10px;color:{'#333' if is_active else '#ccc'}'>{role}</span>"
        )
        annotations.append(dict(
            xref="paper", yref="paper", x=layout["cx"], y=y1 + 0.012,
            yanchor="bottom", showarrow=False, align="center",
            text=(f"<span style='font-size:{title_size}px;font-weight:800;color:{color if is_active else _rgba(color, 0.30)}'>{name}</span>"
                  f"{role_line}"),
        ))

        if mat is not None:
            annotations.append(dict(
                xref="paper", yref="paper", x=layout["cx"], y=y0 - 0.012,
                yanchor="top", showarrow=False, align="center",
                text=(f"<span style='font-size:9px;color:{'#333' if is_active else '#bbb'}'>[{_gqa_shape_label(name, mat)}]</span>"
                      f"<br><span style='font-size:8px;color:{'#666' if is_active else '#ddd'}'>{formula}</span>"),
            ))

    shapes.append(dict(type="line", xref="paper", yref="paper", x0=0.020, x1=0.972,
                       y0=0.480, y1=0.480, line=dict(color="rgba(80,80,80,0.18)", width=1),
                       layer="below"))
    for upper_name, lower_name in (("KG1", "VG1"), ("KG2", "VG2")):
        ux0, uy0, ux1, _ = _gqa_bounds(upper_name)
        lx0, _, lx1, ly1 = _gqa_bounds(lower_name)
        shapes.append(dict(
            type="line", xref="paper", yref="paper",
            x0=min(ux0, lx0) - 0.010, x1=max(ux1, lx1) + 0.010,
            y0=(uy0 + ly1) / 2, y1=(uy0 + ly1) / 2,
            line=dict(color="rgba(17,122,101,0.35)", width=1.2, dash="dot"),
            layer="above",
        ))
    annotations.extend([
        dict(xref="paper", yref="paper", x=0.012, y=0.680,
             text="<b>Group 1</b><br>heads 1-2 share KG1/VG1",
             textangle=-90, showarrow=False, align="center",
             bgcolor="rgba(255,255,255,0.92)", borderpad=3,
             font=dict(size=12, color="#117a65")),
        dict(xref="paper", yref="paper", x=0.012, y=0.280,
             text="<b>Group 2</b><br>heads 3-4 share KG2/VG2",
             textangle=-90, showarrow=False, align="center",
             bgcolor="rgba(255,255,255,0.92)", borderpad=3,
             font=dict(size=12, color="#2874a6")),
    ])

    for src, dst, edge_key, label in GQA_EDGES:
        is_active = edge_key in active_edge_set
        alpha = 0.86 if is_active else 0.09
        width = 2.0 if is_active else 0.7
        sx, sy = _gqa_port(src, "right")
        dx, dy = _gqa_port(dst, "left")
        route = "short" if src.startswith("W") or dst.startswith("W") or src in ("Concat", "Wo") else "scurve"
        path = _bezier_path(sx, sy, dx, dy, route)
        shapes.append(dict(
            type="path", xref="paper", yref="paper", path=path,
            fillcolor="rgba(0,0,0,0)",
            line=dict(color=_rgba("#1c2833", alpha), width=width),
            layer="above",
        ))
        annotations.append(dict(
            xref="paper", yref="paper", x=dx, y=dy,
            ax=-16, ay=0, axref="pixel", ayref="pixel",
            showarrow=True, arrowhead=3, arrowsize=0.85, arrowwidth=width,
            arrowcolor=_rgba("#1c2833", alpha), text="",
        ))
        if label and is_active:
            label_y = max(sy, dy) + 0.018
            annotations.append(dict(
                xref="paper", yref="paper", x=(sx + dx) / 2, y=label_y,
                text=f"<b>{label}</b>", showarrow=False,
                bgcolor="rgba(255,255,255,0.90)", borderpad=2,
                font=dict(size=8, color="#1c2833"),
            ))

    node_list = list(GQA_LAYOUT.keys())
    for name, layout in GQA_LAYOUT.items():
        mat = matrices.get(name)
        if mat is None:
            continue
        is_active = name in active_node_set
        display = mat.copy().astype(float)
        rows, cols = display.shape
        is_param_node = layout["nt"] == "param"
        hover = [[f"<b>{name}[{r},{c}]</b><br>{display[r, c]:.4f}"
                  for c in range(cols)] for r in range(rows)]
        idx = node_list.index(name)
        xax = "x" if idx == 0 else f"x{idx+1}"
        yax = "y" if idx == 0 else f"y{idx+1}"
        xak = "xaxis" if idx == 0 else f"xaxis{idx+1}"
        yak = "yaxis" if idx == 0 else f"yaxis{idx+1}"
        x0, y0, x1, y1 = _gqa_bounds(name)
        row_labels = [tokens[r] if r < len(tokens) else str(r) for r in range(rows)]
        trace = go.Heatmap(
            z=display,
            colorscale="Blues" if name.startswith("A") else ("YlOrBr" if is_param_node else "Viridis"),
            showscale=False,
            opacity=0.92 if is_active else 0.18,
            hoverinfo="text",
            hovertemplate="%{text}<extra></extra>",
            text=hover,
            y=row_labels if name == "X" else None,
            xaxis=xax,
            yaxis=yax,
        )
        traces.append((trace, xak, yak, x0, y0, x1, y1, name))

    if decode_row_view:
        _blank_old_rows(
            shapes,
            _gqa_bounds,
            ("X", "Q1", "Q2", "Q3", "Q4", "A1", "A2", "A3", "A4", "C1", "C2", "C3", "C4", "Concat", "GQAOut"),
            seq_len,
        )

    for node_name in ("X", "Q1", "Q2", "Q3", "Q4", "KG1", "VG1", "KG2", "VG2",
                      "A1", "A2", "A3", "A4", "C1", "C2", "C3", "C4", "Concat", "GQAOut"):
        x0, y0, x1, y1 = _gqa_bounds(node_name)
        band_y0, band_y1 = _row_band((x0, y0, x1, y1), seq_len, new_row)
        shapes.append(dict(
            type="rect", xref="paper", yref="paper",
            x0=x0, y0=band_y0, x1=x1, y1=band_y1,
            fillcolor=_rgba(new_color, 0.34),
            line=dict(color=_rgba(new_color, 0.95), width=2.2),
            layer="above",
        ))

    for row in range(seq_len - 1):
        for node_name in ("KG1", "VG1", "KG2", "VG2"):
            x0, y0, x1, y1 = _gqa_bounds(node_name)
            band_y0, band_y1 = _row_band((x0, y0, x1, y1), seq_len, row)
            shapes.append(dict(
                type="rect", xref="paper", yref="paper",
                x0=x0, y0=band_y0, x1=x1, y1=band_y1,
                fillcolor=_rgba(cached_color, 0.34),
                line=dict(color=_rgba(cached_color, 1.0), width=2.4),
                layer="above",
            ))

    x0, y0, x1, y1 = _gqa_bounds("Concat")
    for i in range(1, 4):
        x = x0 + (x1 - x0) * i / 4
        shapes.append(dict(type="line", xref="paper", yref="paper",
                           x0=x, y0=y0, x1=x, y1=y1,
                           line=dict(color="rgba(255,255,255,0.80)", width=1.4),
                           layer="above"))

    annotations.extend([
        dict(xref="paper", yref="paper", x=0.600, y=1.225,
             text="<span style='color:#d35400'><b>orange</b></span> = newest token rows",
             showarrow=False, xanchor="left", bgcolor="rgba(255,255,255,0.92)",
             borderpad=3, font=dict(size=11, color="#333")),
        dict(xref="paper", yref="paper", x=0.600, y=1.185,
             text="<span style='color:#0057b8'><b>blue</b></span> = cached K/V rows per group",
             showarrow=False, xanchor="left", bgcolor="rgba(255,255,255,0.92)",
             borderpad=3, font=dict(size=11, color="#333")),
        dict(xref="paper", yref="paper", x=0.600, y=1.145,
             text="Q rows are computed for the newest token only; Q is not cached",
             showarrow=False, xanchor="left", bgcolor="rgba(255,255,255,0.92)",
             borderpad=3, font=dict(size=11, color="#333")),
        dict(xref="paper", yref="paper", x=0.600, y=1.105,
             text="Concat newest row = [C1_new | C2_new | C3_new | C4_new]",
             showarrow=False, xanchor="left", bgcolor="rgba(255,255,255,0.92)",
             borderpad=3, font=dict(size=11, color="#333")),
    ])
    if decode_row_view:
        annotations.append(dict(xref="paper", yref="paper", x=0.600, y=1.065,
             text="Decode row view: old non-cache rows are hidden",
             showarrow=False, xanchor="left", bgcolor="rgba(255,255,255,0.92)",
             borderpad=3, font=dict(size=11, color="#333")))

    layout_args = dict(
        paper_bgcolor="white", plot_bgcolor="white",
        width=fig_width, height=fig_height,
        margin=dict(l=8, r=8, t=205, b=56),
        showlegend=False,
        shapes=shapes,
        annotations=annotations,
    )
    for _, xak, yak, x0, y0, x1, y1, name in traces:
        layout_args[xak] = dict(domain=[x0, x1], showticklabels=False,
                                showgrid=False, zeroline=False, fixedrange=True)
        layout_args[yak] = dict(domain=[y0, y1], showticklabels=(name == "X"),
                                tickfont=dict(size=8, color="#444"), side="right",
                                showgrid=False, zeroline=False, fixedrange=True,
                                autorange="reversed")

    return go.Figure(data=[t[0] for t in traces], layout=go.Layout(**layout_args))


def gqa_detail_panel(node_name, matrices, tokens):
    mat = matrices.get(node_name)
    if mat is None:
        return go.Figure()
    display = mat.copy().astype(float)
    rows, cols = display.shape
    role, formula = GQA_ROLES.get(node_name, ("", ""))
    title_text = f"<b>{node_name}</b>  <span style='color:#888'>{role}  [{rows}×{cols}]</span>"
    if formula:
        title_text += f"   <span style='color:#aaa'>{formula}</span>"
    row_labels = [tokens[r] if r < len(tokens) else str(r) for r in range(rows)]
    col_labels = [f"d{c}" for c in range(cols)]
    hover = [[f"{display[r, c]:.4f}" for c in range(cols)] for r in range(rows)]
    fig = go.Figure(go.Heatmap(
        z=display,
        colorscale="Blues" if node_name.startswith("A") else "Viridis",
        showscale=True,
        hovertemplate="%{text}<extra></extra>",
        text=hover,
        x=col_labels,
        y=row_labels,
    ))
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=12)),
        height=270,
        margin=dict(l=50, r=10, t=48, b=30),
        paper_bgcolor="white",
        plot_bgcolor="white",
        yaxis=dict(autorange="reversed"),
    )
    return fig


# ---------------------------------------------------------------------------
# KV cache flow
# ---------------------------------------------------------------------------

KV_CACHE_LAYOUT = {
    "XFull":      dict(cx=0.045, cy=0.55, w=0.060, h=0.170, nt="input"),
    "XNew":       dict(cx=0.160, cy=0.55, w=0.060, h=0.060, nt="new"),
    "q_new":      dict(cx=0.300, cy=0.78, w=0.060, h=0.060, nt="activation"),
    "k_new":      dict(cx=0.300, cy=0.55, w=0.060, h=0.060, nt="activation"),
    "v_new":      dict(cx=0.300, cy=0.32, w=0.060, h=0.060, nt="activation"),
    "K_cache":    dict(cx=0.470, cy=0.68, w=0.070, h=0.135, nt="cache"),
    "V_cache":    dict(cx=0.470, cy=0.40, w=0.070, h=0.135, nt="cache"),
    "K_full":     dict(cx=0.640, cy=0.68, w=0.075, h=0.170, nt="kv"),
    "V_full":     dict(cx=0.640, cy=0.40, w=0.075, h=0.170, nt="kv"),
    "ScoresNew":  dict(cx=0.780, cy=0.78, w=0.085, h=0.060, nt="score"),
    "AttnNew":    dict(cx=0.885, cy=0.78, w=0.085, h=0.060, nt="attn"),
    "ContextNew": dict(cx=0.885, cy=0.40, w=0.075, h=0.060, nt="output"),
}

KV_CACHE_ROLES = {
    "XFull":      ("Input X", "old rows + new row"),
    "XNew":       ("New Token Row", "last row of X"),
    "q_new":      ("New Query", "fresh compute"),
    "k_new":      ("New Key", "append to cache"),
    "v_new":      ("New Value", "append to cache"),
    "K_cache":    ("K Cache", "reused old rows"),
    "V_cache":    ("V Cache", "reused old rows"),
    "K_full":     ("Full K", "[K_cache; k_new]"),
    "V_full":     ("Full V", "[V_cache; v_new]"),
    "ScoresNew":  ("New Score Row", "q_new @ K_full.T"),
    "AttnNew":    ("New Attention Row", "softmax"),
    "ContextNew": ("New Context", "AttnNew @ V_full"),
}

_KV_CACHE_COLOR = {
    "input": "#7d3c98",
    "new": "#d35400",
    "activation": "#c0392b",
    "cache": "#2874a6",
    "kv": "#117a65",
    "score": "#1a5276",
    "attn": "#117a65",
    "output": "#a04000",
}

KV_CACHE_EDGES = [
    ("XFull", "XNew", "XFull->XNew", "take new row"),
    ("XNew", "q_new", "XNew->q_new", "Wq"),
    ("XNew", "k_new", "XNew->k_new", "Wk"),
    ("XNew", "v_new", "XNew->v_new", "Wv"),
    ("K_cache", "K_full", "K_cache->K_full", "cached rows"),
    ("k_new", "K_full", "k_new->K_full", "append"),
    ("V_cache", "V_full", "V_cache->V_full", "cached rows"),
    ("v_new", "V_full", "v_new->V_full", "append"),
    ("q_new", "ScoresNew", "q_new->ScoresNew", ""),
    ("K_full", "ScoresNew", "K_full->ScoresNew", "q_new · Kᵀ"),
    ("ScoresNew", "AttnNew", "ScoresNew->AttnNew", "softmax"),
    ("AttnNew", "ContextNew", "AttnNew->ContextNew", ""),
    ("V_full", "ContextNew", "V_full->ContextNew", "A · V"),
]


def _kv_bounds(name):
    n = KV_CACHE_LAYOUT[name]
    hw, hh = n["w"] / 2, n["h"] / 2
    return n["cx"] - hw, n["cy"] - hh, n["cx"] + hw, n["cy"] + hh


def _kv_port(name, side, y_frac=0.5):
    x0, y0, x1, y1 = _kv_bounds(name)
    py = y0 + y_frac * (y1 - y0)
    return (x1, py) if side == "right" else (x0, py)


def _kv_shape_label(mat):
    rows, cols = mat.shape
    return f"{rows}×{cols}"


def _row_band(bounds, rows, row_index):
    x0, y0, x1, y1 = bounds
    row_h = (y1 - y0) / rows
    top = y1 - row_index * row_h
    return top - row_h, top


def _blank_old_rows(shapes, bounds_fn, node_names, rows):
    if rows <= 1:
        return
    for node_name in node_names:
        x0, y0, x1, y1 = bounds_fn(node_name)
        for row in range(rows - 1):
            band_y0, band_y1 = _row_band((x0, y0, x1, y1), rows, row)
            shapes.append(dict(
                type="rect", xref="paper", yref="paper",
                x0=x0, y0=band_y0, x1=x1, y1=band_y1,
                fillcolor="rgba(255,255,255,0.86)",
                line=dict(color="rgba(255,255,255,0)", width=0),
                layer="above",
            ))


def _add_row_overlay(shapes, name, rows, row_index, color, alpha):
    x0, y0, x1, y1 = _kv_bounds(name)
    band_y0, band_y1 = _row_band((x0, y0, x1, y1), rows, row_index)
    shapes.append(dict(
        type="rect", xref="paper", yref="paper",
        x0=x0, y0=band_y0, x1=x1, y1=band_y1,
        fillcolor=_rgba(color, alpha),
        line=dict(color=_rgba(color, min(alpha + 0.25, 0.95)), width=1.0),
        layer="above",
    ))


def kv_cache_diagram(matrices, active_nodes, active_edges, tokens,
                     fig_width=1500, fig_height=650):
    active_node_set = set(active_nodes)
    active_edge_set = set(active_edges)
    shapes = []
    annotations = []
    traces = []

    for name, layout in KV_CACHE_LAYOUT.items():
        is_active = name in active_node_set
        color = _KV_CACHE_COLOR[layout["nt"]]
        x0, y0, x1, y1 = _kv_bounds(name)
        shapes.append(dict(
            type="rect", xref="paper", yref="paper",
            x0=x0, y0=y0, x1=x1, y1=y1,
            fillcolor=_rgba(color, 0.12 if is_active else 0.035),
            line=dict(color=_rgba(color, 1.0 if is_active else 0.18),
                      width=2.4 if is_active else 1.0),
            layer="below",
        ))

        role, formula = KV_CACHE_ROLES[name]
        annotations.append(dict(
            xref="paper", yref="paper", x=layout["cx"], y=y1 + 0.024,
            yanchor="bottom", showarrow=False, align="center",
            text=(f"<span style='font-size:17px;font-weight:800;color:{color if is_active else _rgba(color, 0.30)}'>{name}</span>"
                  f"<br><span style='font-size:11px;color:{'#333' if is_active else '#ccc'}'>{role}</span>"),
        ))

        mat = matrices.get(name)
        if mat is not None:
            annotations.append(dict(
                xref="paper", yref="paper", x=layout["cx"], y=y0 - 0.018,
                yanchor="top", showarrow=False, align="center",
                text=(f"<span style='font-size:10px;color:{'#333' if is_active else '#bbb'}'>[{_kv_shape_label(mat)}]</span>"
                      f"<br><span style='font-size:9px;color:{'#666' if is_active else '#ddd'}'>{formula}</span>"),
            ))

    for src, dst, edge_key, label in KV_CACHE_EDGES:
        is_active = edge_key in active_edge_set
        alpha = 0.86 if is_active else 0.10
        width = 2.0 if is_active else 0.7
        sx, sy = _kv_port(src, "right")
        dx, dy = _kv_port(dst, "left")
        path = _bezier_path(sx, sy, dx, dy, "scurve")
        shapes.append(dict(
            type="path", xref="paper", yref="paper", path=path,
            fillcolor="rgba(0,0,0,0)",
            line=dict(color=_rgba("#1c2833", alpha), width=width),
            layer="above",
        ))
        annotations.append(dict(
            xref="paper", yref="paper", x=dx, y=dy,
            ax=-16, ay=0, axref="pixel", ayref="pixel",
            showarrow=True, arrowhead=3, arrowsize=0.85, arrowwidth=width,
            arrowcolor=_rgba("#1c2833", alpha), text="",
        ))
        if label and is_active:
            annotations.append(dict(
                xref="paper", yref="paper", x=(sx + dx) / 2, y=max(sy, dy) + 0.035,
                text=f"<b>{label}</b>", showarrow=False,
                bgcolor="rgba(255,255,255,0.90)", borderpad=2,
                font=dict(size=9, color="#1c2833"),
            ))

    node_list = list(KV_CACHE_LAYOUT.keys())
    for name, layout in KV_CACHE_LAYOUT.items():
        mat = matrices.get(name)
        if mat is None:
            continue
        is_active = name in active_node_set
        display = mat.copy().astype(float)
        rows, cols = display.shape
        hover = [[f"<b>{name}[{r},{c}]</b><br>{display[r, c]:.4f}"
                  for c in range(cols)] for r in range(rows)]
        idx = node_list.index(name)
        xax = "x" if idx == 0 else f"x{idx+1}"
        yax = "y" if idx == 0 else f"y{idx+1}"
        xak = "xaxis" if idx == 0 else f"xaxis{idx+1}"
        yak = "yaxis" if idx == 0 else f"yaxis{idx+1}"
        x0, y0, x1, y1 = _kv_bounds(name)
        row_labels = [tokens[r] if r < len(tokens) else str(r) for r in range(rows)]
        if rows == 1:
            row_labels = ["new"]
        trace = go.Heatmap(
            z=display,
            colorscale="Blues" if layout["nt"] == "cache" else ("Oranges" if layout["nt"] == "new" else "Viridis"),
            showscale=False,
            opacity=0.92 if is_active else 0.18,
            hoverinfo="text",
            hovertemplate="%{text}<extra></extra>",
            text=hover,
            y=row_labels if name in ("XFull", "K_full", "V_full") else None,
            xaxis=xax,
            yaxis=yax,
        )
        traces.append((trace, xak, yak, x0, y0, x1, y1, name))

    n = matrices["XFull"].shape[0]
    prev_n = max(n - 1, 1)
    _add_row_overlay(shapes, "XFull", n, n - 1, "#d35400", 0.28)
    _add_row_overlay(shapes, "K_full", n, n - 1, "#d35400", 0.28)
    _add_row_overlay(shapes, "V_full", n, n - 1, "#d35400", 0.28)
    for row in range(n - 1):
        _add_row_overlay(shapes, "K_full", n, row, "#2874a6", 0.10)
        _add_row_overlay(shapes, "V_full", n, row, "#2874a6", 0.10)
    for row in range(prev_n):
        _add_row_overlay(shapes, "K_cache", prev_n, row, "#2874a6", 0.12)
        _add_row_overlay(shapes, "V_cache", prev_n, row, "#2874a6", 0.12)

    annotations.extend([
        dict(xref="paper", yref="paper", x=0.055, y=0.255,
             text="<span style='color:#d35400'><b>orange</b></span> = newly added token row",
             showarrow=False, font=dict(size=11, color="#333")),
        dict(xref="paper", yref="paper", x=0.245, y=0.255,
             text="<span style='color:#2874a6'><b>blue</b></span> = reused from KV cache",
             showarrow=False, font=dict(size=11, color="#333")),
    ])

    layout_args = dict(
        paper_bgcolor="white", plot_bgcolor="white",
        width=fig_width, height=fig_height,
        margin=dict(l=8, r=8, t=70, b=58),
        showlegend=False,
        shapes=shapes,
        annotations=annotations,
    )
    for _, xak, yak, x0, y0, x1, y1, name in traces:
        layout_args[xak] = dict(
            domain=[x0, x1], showticklabels=False,
            showgrid=False, zeroline=False, fixedrange=True,
        )
        layout_args[yak] = dict(
            domain=[y0, y1], showticklabels=name in ("XFull", "K_full", "V_full"),
            tickfont=dict(size=8, color="#444"),
            side="right",
            showgrid=False, zeroline=False, fixedrange=True,
            autorange="reversed",
        )

    return go.Figure(data=[t[0] for t in traces], layout=go.Layout(**layout_args))


def kv_cache_detail_panel(node_name, matrices, tokens):
    mat = matrices.get(node_name)
    if mat is None:
        return go.Figure()
    display = mat.copy().astype(float)
    rows, cols = display.shape
    role, formula = KV_CACHE_ROLES.get(node_name, ("", ""))
    title_text = f"<b>{node_name}</b>  <span style='color:#888'>{role}  [{rows}×{cols}]</span>"
    if formula:
        title_text += f"   <span style='color:#aaa'>{formula}</span>"
    row_labels = [tokens[r] if r < len(tokens) else str(r) for r in range(rows)]
    if rows == 1:
        row_labels = ["new"]
    col_labels = [f"d{c}" for c in range(cols)]
    hover = [[f"{display[r, c]:.4f}" for c in range(cols)] for r in range(rows)]
    fig = go.Figure(go.Heatmap(
        z=display,
        colorscale="Viridis",
        showscale=True,
        hovertemplate="%{text}<extra></extra>",
        text=hover,
        x=col_labels,
        y=row_labels,
    ))
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=12)),
        height=270,
        margin=dict(l=50, r=10, t=48, b=30),
        paper_bgcolor="white",
        plot_bgcolor="white",
        yaxis=dict(autorange="reversed"),
    )
    return fig


# Override the earlier cache-specific renderer with the same matrix layout as
# the main attention flow, plus row overlays for cached vs newly added values.
KV_CACHE_LAYOUT = NODE_LAYOUT


def _add_attention_row_overlay(shapes, node_name, rows, row_index, color, alpha):
    x0, y0, x1, y1 = _node_bounds(node_name)
    band_y0, band_y1 = _row_band((x0, y0, x1, y1), rows, row_index)
    shapes.append(dict(
        type="rect", xref="paper", yref="paper",
        x0=x0, y0=band_y0, x1=x1, y1=band_y1,
        fillcolor=_rgba(color, alpha),
        line=dict(color=_rgba(color, 0.95), width=2.2),
        layer="above",
    ))


def kv_cache_diagram(matrices, active_nodes, active_edges,
                     seq_len, d_model, d_head, tokens,
                     decode_row_view=False,
                     fig_width=1400, fig_height=700):
    fig = flow_diagram(
        matrices=matrices,
        active_nodes=active_nodes,
        active_edges=active_edges,
        seq_len=seq_len,
        d_model=d_model,
        d_head=d_head,
        tokens=tokens,
        fig_width=fig_width,
        fig_height=fig_height,
    )

    shapes = list(fig.layout.shapes)
    annotations = list(fig.layout.annotations)
    cached_color = "#0057b8"
    new_color = "#d35400"
    new_row = seq_len - 1

    if decode_row_view:
        _blank_old_rows(shapes, _node_bounds, ("X", "Q", "S", "S_scaled", "S_masked", "A", "C"), seq_len)

    # New token-derived rows.
    for node_name in ("X", "Q", "K", "V", "C"):
        _add_attention_row_overlay(shapes, node_name, seq_len, new_row, new_color, 0.30)
    for node_name in ("S", "S_scaled", "S_masked", "A"):
        _add_attention_row_overlay(shapes, node_name, seq_len, new_row, new_color, 0.42)

    # The reusable cache is specifically previous K/V rows.
    for row in range(seq_len - 1):
        _add_attention_row_overlay(shapes, "K", seq_len, row, cached_color, 0.34)
        _add_attention_row_overlay(shapes, "V", seq_len, row, cached_color, 0.34)

    annotations.extend([
        dict(
            xref="paper", yref="paper", x=0.785, y=0.940,
            text="<span style='color:#d35400'><b>orange</b></span> = values added for the newest token",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.92)",
            borderpad=3,
            font=dict(size=11, color="#333"),
        ),
        dict(
            xref="paper", yref="paper", x=0.785, y=0.900,
            text="<span style='color:#0057b8'><b>blue</b></span> = previous K/V rows reused from cache",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.92)",
            borderpad=3,
            font=dict(size=11, color="#333"),
        ),
        dict(
            xref="paper", yref="paper", x=0.785, y=0.860,
            text="Q row is computed for the newest token only; Q is not cached",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.92)",
            borderpad=3,
            font=dict(size=11, color="#333"),
        ),
        dict(
            xref="paper", yref="paper", x=0.785, y=0.820,
            text="Only newest C row continues to next-token logits",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.92)",
            borderpad=3,
            font=dict(size=11, color="#333"),
        ),
    ])
    if decode_row_view:
        annotations.append(dict(
            xref="paper", yref="paper", x=0.785, y=0.780,
            text="Decode row view: old non-cache rows are hidden",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.92)",
            borderpad=3,
            font=dict(size=11, color="#333"),
        ))

    fig.update_layout(shapes=shapes, annotations=annotations)
    return fig


# ---------------------------------------------------------------------------
# Multi-head latent attention KV cache flow
# ---------------------------------------------------------------------------

MLA_LAYOUT = {
    "X":        dict(cx=0.045, cy=0.52, w=0.060, h=0.120, nt="input"),
    "Wq":       dict(cx=0.165, cy=0.80, w=0.055, h=0.080, nt="param", param=True),
    "Q":        dict(cx=0.275, cy=0.80, w=0.060, h=0.095, nt="activation"),
    "Wdkv":     dict(cx=0.165, cy=0.52, w=0.060, h=0.080, nt="param", param=True),
    "cKV":      dict(cx=0.300, cy=0.52, w=0.052, h=0.120, nt="latent"),
    "Wuk":      dict(cx=0.425, cy=0.68, w=0.055, h=0.080, nt="param", param=True),
    "K":        dict(cx=0.525, cy=0.68, w=0.060, h=0.105, nt="activation"),
    "Wuv":      dict(cx=0.425, cy=0.34, w=0.055, h=0.080, nt="param", param=True),
    "V":        dict(cx=0.525, cy=0.34, w=0.060, h=0.105, nt="activation"),
    "S":        dict(cx=0.635, cy=0.68, w=0.052, h=0.095, nt="score"),
    "S_scaled": dict(cx=0.720, cy=0.68, w=0.052, h=0.095, nt="score"),
    "S_masked": dict(cx=0.805, cy=0.68, w=0.052, h=0.095, nt="score"),
    "A":        dict(cx=0.890, cy=0.68, w=0.052, h=0.095, nt="attn"),
    "C":        dict(cx=0.855, cy=0.34, w=0.060, h=0.105, nt="output"),
    "Wo":       dict(cx=0.950, cy=0.34, w=0.055, h=0.080, nt="param", param=True),
}

MLA_ROLES = {
    "X": ("Input", "old rows + newest row"),
    "Wq": ("Query Weight", "learned param"),
    "Q": ("Queries", "= X @ Wq"),
    "Wdkv": ("KV Down-Projector", "learned compress"),
    "cKV": ("Latent KV Cache", "only cached matrix"),
    "Wuk": ("Key Up-Projector", "learned decompress"),
    "K": ("Reconstructed Keys", "= cKV @ Wuk"),
    "Wuv": ("Value Up-Projector", "learned decompress"),
    "V": ("Reconstructed Values", "= cKV @ Wuv"),
    "S": ("Raw Scores", "= Q @ K.T"),
    "S_scaled": ("Scaled Scores", "= S / sqrt(d)"),
    "S_masked": ("Masked Scores", "causal mask"),
    "A": ("Attn Weights", "= softmax"),
    "C": ("Context Output", "= A @ V"),
    "Wo": ("Output Weight", "learned param"),
}

_MLA_COLOR = {
    "input": "#7d3c98",
    "param": "#b7770d",
    "activation": "#c0392b",
    "latent": "#0057b8",
    "score": "#1a5276",
    "attn": "#117a65",
    "output": "#a04000",
}

MLA_EDGES = [
    ("X", "Wq", "X->Wq", ""),
    ("Wq", "Q", "Wq->Q", "X · Wq"),
    ("X", "Wdkv", "X->Wdkv", ""),
    ("Wdkv", "cKV", "Wdkv->cKV", "X · Wdkv"),
    ("cKV", "Wuk", "cKV->Wuk", ""),
    ("Wuk", "K", "Wuk->K", "cKV left · Wuk right"),
    ("cKV", "Wuv", "cKV->Wuv", ""),
    ("Wuv", "V", "Wuv->V", "cKV left · Wuv right"),
    ("Q", "S", "Q->S", "Q · Kᵀ"),
    ("K", "S", "K->S", ""),
    ("S", "S_scaled", "S->S_scaled", "÷ √d"),
    ("S_scaled", "S_masked", "S_scaled->S_masked", "causal mask"),
    ("S_masked", "A", "S_masked->A", "softmax"),
    ("A", "C", "A->C", "A · V"),
    ("V", "C", "V->C", ""),
    ("C", "Wo", "C->Wo", "C · Wo"),
]


def _mla_bounds(name):
    n = MLA_LAYOUT[name]
    hw, hh = n["w"] / 2, n["h"] / 2
    return n["cx"] - hw, n["cy"] - hh, n["cx"] + hw, n["cy"] + hh


def _mla_port(name, side, y_frac=0.5):
    x0, y0, x1, y1 = _mla_bounds(name)
    py = y0 + y_frac * (y1 - y0)
    return (x1, py) if side == "right" else (x0, py)


def _mla_shape_label(name, mat):
    rows, cols = mat.shape
    return f"{rows}×{cols}"


def _add_mla_row_overlay(shapes, node_name, rows, row_index, color, alpha):
    x0, y0, x1, y1 = _mla_bounds(node_name)
    band_y0, band_y1 = _row_band((x0, y0, x1, y1), rows, row_index)
    shapes.append(dict(
        type="rect", xref="paper", yref="paper",
        x0=x0, y0=band_y0, x1=x1, y1=band_y1,
        fillcolor=_rgba(color, alpha),
        line=dict(color=_rgba(color, 0.95), width=2.2),
        layer="above",
    ))


MLA_ABS_LAYOUT = {
    "XNew":       dict(cx=0.060, cy=0.60, w=0.070, h=0.060, nt="input"),
    "Wscore":     dict(cx=0.210, cy=0.60, w=0.085, h=0.090, nt="param", param=True),
    "cKV":        dict(cx=0.365, cy=0.60, w=0.060, h=0.150, nt="latent"),
    "S_abs":      dict(cx=0.520, cy=0.60, w=0.085, h=0.060, nt="score"),
    "A_abs":      dict(cx=0.650, cy=0.60, w=0.085, h=0.060, nt="attn"),
    "Wout":       dict(cx=0.650, cy=0.32, w=0.085, h=0.090, nt="param", param=True),
    "MLAOut_abs": dict(cx=0.825, cy=0.46, w=0.085, h=0.060, nt="output"),
}

MLA_ABS_ROLES = {
    "XNew": ("Newest Input Row", "x_new"),
    "Wscore": ("Folded Score Weight", "Wq @ Wuk.T"),
    "cKV": ("Latent KV Cache", "cached history"),
    "S_abs": ("New Score Row", "x_new @ Wscore @ cKV.T"),
    "A_abs": ("New Attention Row", "softmax"),
    "Wout": ("Folded Output Weight", "Wuv @ Wo"),
    "MLAOut_abs": ("New Output Row", "A_new @ cKV @ Wout"),
}

MLA_ABS_LEGEND = {
    "XNew": "x_new = newest row of X",
    "Wscore": "Wscore = Wq @ Wuk.T",
    "cKV": "cKV = X @ Wdkv; cache stores this",
    "S_abs": "S_new = x_new @ Wscore @ cKV.T",
    "A_abs": "A_new = softmax(S_new / sqrt(d_head))",
    "Wout": "Wout = Wuv @ Wo",
    "MLAOut_abs": "output_new = A_new @ cKV @ Wout",
}

MLA_ABS_EDGES = [
    ("XNew", "Wscore", "x_new"),
    ("Wscore", "S_abs", ""),
    ("cKV", "S_abs", "cKV.T"),
    ("S_abs", "A_abs", "softmax"),
    ("A_abs", "MLAOut_abs", ""),
    ("cKV", "MLAOut_abs", "cKV"),
    ("Wout", "MLAOut_abs", ""),
]


def _mla_abs_bounds(name):
    n = MLA_ABS_LAYOUT[name]
    hw, hh = n["w"] / 2, n["h"] / 2
    return n["cx"] - hw, n["cy"] - hh, n["cx"] + hw, n["cy"] + hh


def _mla_abs_port(name, side, y_frac=0.5):
    x0, y0, x1, y1 = _mla_abs_bounds(name)
    py = y0 + y_frac * (y1 - y0)
    return (x1, py) if side == "right" else (x0, py)


def _abs_trace_for(name, matrices, active, tokens, node_list):
    mat = matrices.get(name)
    if mat is None:
        return None
    layout = MLA_ABS_LAYOUT[name]
    display = mat.copy().astype(float)
    rows, cols = display.shape
    idx = node_list.index(name)
    xax = "x" if idx == 0 else f"x{idx+1}"
    yax = "y" if idx == 0 else f"y{idx+1}"
    xak = "xaxis" if idx == 0 else f"xaxis{idx+1}"
    yak = "yaxis" if idx == 0 else f"yaxis{idx+1}"
    x0, y0, x1, y1 = _mla_abs_bounds(name)
    hover = [[f"<b>{name}[{r},{c}]</b><br>{display[r, c]:.4f}" for c in range(cols)] for r in range(rows)]
    row_labels = ["new"] if rows == 1 else [tokens[r] if r < len(tokens) else str(r) for r in range(rows)]
    colorscale = "YlOrBr" if layout["nt"] == "param" else ("Blues" if layout["nt"] in ("score", "attn") else "Viridis")
    return (
        go.Heatmap(
            z=display,
            colorscale=colorscale,
            showscale=False,
            opacity=0.94 if active else 0.22,
            hoverinfo="text",
            hovertemplate="%{text}<extra></extra>",
            text=hover,
            y=row_labels if name in ("XNew", "cKV") else None,
            xaxis=xax,
            yaxis=yax,
        ),
        xak, yak, x0, y0, x1, y1, name,
    )


def mla_absorption_diagram(matrices, active_nodes, active_edges, tokens,
                           fig_width=1500, fig_height=620):
    active_node_set = set(active_nodes)
    shapes = []
    annotations = []
    traces = []
    cached_color = "#0057b8"
    new_color = "#d35400"

    for name, layout in MLA_ABS_LAYOUT.items():
        is_active = name in active_node_set or name in ("XNew", "Wscore", "cKV", "S_abs", "A_abs", "Wout", "MLAOut_abs")
        color = _MLA_COLOR[layout["nt"]]
        x0, y0, x1, y1 = _mla_abs_bounds(name)
        shapes.append(dict(
            type="rect", xref="paper", yref="paper",
            x0=x0, y0=y0, x1=x1, y1=y1,
            fillcolor=_rgba(color, 0.12),
            line=dict(color=_rgba(color, 1.0), width=2.4, dash="dash" if layout.get("param") else "solid"),
            layer="below",
        ))
        role, formula = MLA_ABS_ROLES[name]
        mat = matrices.get(name)
        shape_label = f"[{mat.shape[0]}×{mat.shape[1]}]" if mat is not None else ""
        annotations.append(dict(
            xref="paper", yref="paper", x=layout["cx"], y=y1 + 0.022,
            yanchor="bottom", showarrow=False, align="center",
            text=(f"<span style='font-size:18px;font-weight:800;color:{color}'>{name}</span>"
                  f"<br><span style='font-size:11px;color:#333'>{role}</span>"),
        ))
        annotations.append(dict(
            xref="paper", yref="paper", x=layout["cx"], y=y0 - 0.020,
            yanchor="top", showarrow=False, align="center",
            text=(f"<span style='font-size:10px;color:#333'>{shape_label}</span>"
                  f"<br><span style='font-size:9px;color:#666'>{formula}</span>"),
        ))

    for src, dst, label in MLA_ABS_EDGES:
        sx, sy = _mla_abs_port(src, "right")
        dx, dy = _mla_abs_port(dst, "left")
        route = "short" if src in ("XNew", "Wscore", "S_abs", "A_abs") else "scurve"
        path = _bezier_path(sx, sy, dx, dy, route)
        shapes.append(dict(
            type="path", xref="paper", yref="paper", path=path,
            fillcolor="rgba(0,0,0,0)",
            line=dict(color=_rgba("#1c2833", 0.86), width=2.0),
            layer="above",
        ))
        annotations.append(dict(
            xref="paper", yref="paper", x=dx, y=dy,
            ax=-16, ay=0, axref="pixel", ayref="pixel",
            showarrow=True, arrowhead=3, arrowsize=0.85, arrowwidth=2.0,
            arrowcolor=_rgba("#1c2833", 0.86), text="",
        ))
        if label:
            annotations.append(dict(
                xref="paper", yref="paper", x=(sx + dx) / 2, y=max(sy, dy) + 0.035,
                text=f"<b>{label}</b>", showarrow=False,
                bgcolor="rgba(255,255,255,0.92)", borderpad=2,
                font=dict(size=10, color="#1c2833"),
            ))

    node_list = list(MLA_ABS_LAYOUT.keys())
    for name in node_list:
        trace_info = _abs_trace_for(name, matrices, True, tokens, node_list)
        if trace_info:
            traces.append(trace_info)

    seq_len = matrices["cKV"].shape[0]
    c_x0, c_y0, c_x1, c_y1 = _mla_abs_bounds("cKV")
    for row in range(seq_len - 1):
        band_y0, band_y1 = _row_band((c_x0, c_y0, c_x1, c_y1), seq_len, row)
        shapes.append(dict(
            type="rect", xref="paper", yref="paper",
            x0=c_x0, y0=band_y0, x1=c_x1, y1=band_y1,
            fillcolor=_rgba(cached_color, 0.34),
            line=dict(color=_rgba(cached_color, 1.0), width=2.2),
            layer="above",
        ))
    band_y0, band_y1 = _row_band((c_x0, c_y0, c_x1, c_y1), seq_len, seq_len - 1)
    shapes.append(dict(
        type="rect", xref="paper", yref="paper",
        x0=c_x0, y0=band_y0, x1=c_x1, y1=band_y1,
        fillcolor=_rgba(new_color, 0.32),
        line=dict(color=_rgba(new_color, 0.95), width=2.2),
        layer="above",
    ))

    annotations.extend([
        dict(
            xref="paper", yref="paper", x=0.500, y=0.925,
            text=(
                "<span style='font-size:18px;font-weight:800;color:#1c2833'>Absorbed MLA inference path</span>"
                "<br><span style='font-size:14px;color:#1c2833'>No explicit Q, K, V, or C matrices are materialized in this fused view.</span>"
            ),
            showarrow=False,
            bgcolor="rgba(255,255,255,0.96)",
            borderpad=5,
            align="center",
        ),
        dict(
            xref="paper", yref="paper", x=0.985, y=0.900,
            xanchor="right", yanchor="top",
            text=(
                "<span style='font-size:15px;font-weight:800;color:#1c2833'>Algebra legend</span>"
                + "".join(
                    f"<br><span style='font-size:11px;color:{_MLA_COLOR[MLA_ABS_LAYOUT[name]['nt']]}'><b>{name}</b></span>"
                    f"<span style='font-size:11px;color:#333'>: {definition}</span>"
                    for name, definition in MLA_ABS_LEGEND.items()
                )
            ),
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.96)",
            bordercolor="rgba(28,40,51,0.18)",
            borderwidth=1,
            borderpad=6,
        ),
        dict(
            xref="paper", yref="paper", x=0.500, y=0.075,
            text=(
                "<span style='font-size:15px;color:#1a5276'><b>scores</b></span>"
                "<span style='font-size:15px;color:#1c2833'> = </span>"
                "<span style='font-size:15px;color:#c0392b'>x_new</span>"
                "<span style='font-size:15px;color:#1c2833'> @ </span>"
                "<span style='font-size:15px;color:#b7770d'>(Wq @ Wuk.T)</span>"
                "<span style='font-size:15px;color:#1c2833'> @ </span>"
                "<span style='font-size:15px;color:#0057b8'>cKV.T</span>"
                "<span style='font-size:15px;color:#1c2833'>    |    </span>"
                "<span style='font-size:15px;color:#a04000'><b>output</b></span>"
                "<span style='font-size:15px;color:#1c2833'> = </span>"
                "<span style='font-size:15px;color:#117a65'>A_new</span>"
                "<span style='font-size:15px;color:#1c2833'> @ </span>"
                "<span style='font-size:15px;color:#0057b8'>cKV</span>"
                "<span style='font-size:15px;color:#1c2833'> @ </span>"
                "<span style='font-size:15px;color:#b7770d'>(Wuv @ Wo)</span>"
            ),
            showarrow=False,
            bgcolor="rgba(255,255,255,0.96)",
            bordercolor="rgba(28,40,51,0.18)",
            borderwidth=1,
            borderpad=7,
            align="center",
        ),
    ])

    layout_args = dict(
        paper_bgcolor="white", plot_bgcolor="white",
        width=fig_width, height=fig_height,
        margin=dict(l=8, r=8, t=80, b=80),
        showlegend=False,
        shapes=shapes,
        annotations=annotations,
    )
    for _, xak, yak, x0, y0, x1, y1, name in traces:
        layout_args[xak] = dict(domain=[x0, x1], showticklabels=False,
                                showgrid=False, zeroline=False, fixedrange=True)
        layout_args[yak] = dict(domain=[y0, y1], showticklabels=name in ("XNew", "cKV"),
                                tickfont=dict(size=8, color="#444"), side="right",
                                showgrid=False, zeroline=False, fixedrange=True,
                                autorange="reversed")
    return go.Figure(data=[t[0] for t in traces], layout=go.Layout(**layout_args))


def mla_kv_cache_diagram(matrices, active_nodes, active_edges, tokens,
                         show_absorption=False,
                         decode_row_view=False,
                         fig_width=1500, fig_height=780):
    if show_absorption:
        return mla_absorption_diagram(matrices, active_nodes, active_edges, tokens)

    active_node_set = set(active_nodes)
    active_edge_set = set(active_edges)
    seq_len = matrices["X"].shape[0]
    new_row = seq_len - 1
    cached_color = "#0057b8"
    new_color = "#d35400"
    shapes = []
    annotations = []
    traces = []

    for name, layout in MLA_LAYOUT.items():
        is_active = name in active_node_set
        color = _MLA_COLOR[layout["nt"]]
        has_params = layout.get("param", False)
        x0, y0, x1, y1 = _mla_bounds(name)
        shapes.append(dict(
            type="rect", xref="paper", yref="paper",
            x0=x0, y0=y0, x1=x1, y1=y1,
            fillcolor=_rgba(color, 0.12 if is_active else 0.035),
            line=dict(
                color=_rgba(color, 1.0 if is_active else 0.18),
                width=2.4 if is_active else 1.0,
                dash="dash" if has_params else "solid",
            ),
            layer="below",
        ))

        role, formula = MLA_ROLES[name]
        title_size = 17 if layout["nt"] not in ("param",) else 15
        annotations.append(dict(
            xref="paper", yref="paper", x=layout["cx"], y=y1 + 0.017,
            yanchor="bottom", showarrow=False, align="center",
            text=(f"<span style='font-size:{title_size}px;font-weight:800;color:{color if is_active else _rgba(color, 0.30)}'>{name}</span>"
                  f"<br><span style='font-size:10px;color:{'#333' if is_active else '#ccc'}'>{role}</span>"),
        ))

        mat = matrices.get(name)
        if mat is not None:
            annotations.append(dict(
                xref="paper", yref="paper", x=layout["cx"], y=y0 - 0.016,
                yanchor="top", showarrow=False, align="center",
                text=(f"<span style='font-size:9px;color:{'#333' if is_active else '#bbb'}'>[{_mla_shape_label(name, mat)}]</span>"
                      f"<br><span style='font-size:8px;color:{'#666' if is_active else '#ddd'}'>{formula}</span>"),
            ))

    for src, dst, edge_key, label in MLA_EDGES:
        is_active = edge_key in active_edge_set
        alpha = 0.86 if is_active else 0.10
        width = 2.0 if is_active else 0.7
        sx, sy = _mla_port(src, "right")
        dx, dy = _mla_port(dst, "left")
        route = "short" if src.startswith("W") or dst.startswith("W") or src in ("S", "S_scaled", "S_masked") else "scurve"
        path = _bezier_path(sx, sy, dx, dy, route)
        shapes.append(dict(
            type="path", xref="paper", yref="paper", path=path,
            fillcolor="rgba(0,0,0,0)",
            line=dict(color=_rgba("#1c2833", alpha), width=width),
            layer="above",
        ))
        annotations.append(dict(
            xref="paper", yref="paper", x=dx, y=dy,
            ax=-16, ay=0, axref="pixel", ayref="pixel",
            showarrow=True, arrowhead=3, arrowsize=0.85, arrowwidth=width,
            arrowcolor=_rgba("#1c2833", alpha), text="",
        ))
        if label and is_active:
            annotations.append(dict(
                xref="paper", yref="paper", x=(sx + dx) / 2, y=max(sy, dy) + 0.030,
                text=f"<b>{label}</b>", showarrow=False,
                bgcolor="rgba(255,255,255,0.90)", borderpad=2,
                font=dict(size=9, color="#1c2833"),
            ))

    if show_absorption:
        absorption_groups = [
            (("Wq", "Wuk"), "fold fixed product: Wq · Wukᵀ", 0.034),
            (("Wuv", "Wo"), "fold fixed product: Wuv · Wo", -0.052),
        ]
        for group_nodes, label, label_offset in absorption_groups:
            bounds = [_mla_bounds(node_name) for node_name in group_nodes]
            x0 = min(b[0] for b in bounds) - 0.012
            y0 = min(b[1] for b in bounds) - 0.014
            x1 = max(b[2] for b in bounds) + 0.012
            y1 = max(b[3] for b in bounds) + 0.014
            shapes.append(dict(
                type="rect", xref="paper", yref="paper",
                x0=x0, y0=y0, x1=x1, y1=y1,
                fillcolor="rgba(0,0,0,0)",
                line=dict(color=_rgba("#b7770d", 0.90), width=2.0, dash="dash"),
                layer="above",
            ))
            annotations.append(dict(
                xref="paper", yref="paper",
                x=(x0 + x1) / 2,
                y=(y1 + label_offset if label_offset > 0 else y0 + label_offset),
                text=f"<span style='font-size:13px;font-weight:800;color:#b7770d'>{label}</span>",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.94)",
                borderpad=2,
            ))

        for node_name in ("Q", "K", "V", "C"):
            x0, y0, x1, y1 = _mla_bounds(node_name)
            pad = 0.006
            shapes.extend([
                dict(
                    type="line", xref="paper", yref="paper",
                    x0=x0 - pad, y0=y0 - pad, x1=x1 + pad, y1=y1 + pad,
                    line=dict(color=_rgba("#c0392b", 0.95), width=3.0),
                    layer="above",
                ),
                dict(
                    type="line", xref="paper", yref="paper",
                    x0=x0 - pad, y0=y1 + pad, x1=x1 + pad, y1=y0 - pad,
                    line=dict(color=_rgba("#c0392b", 0.95), width=3.0),
                    layer="above",
                ),
            ])

        annotations.extend([
            dict(
                xref="paper", yref="paper", x=0.515, y=0.835,
                text=(
                    "<span style='font-size:13px;font-weight:800;color:#1c2833'>transpose exposes Wukᵀ</span>"
                    "<br><span style='font-size:12px;color:#0057b8'>(cKV·Wuk)ᵀ</span>"
                    "<span style='font-size:12px;color:#1c2833'> = </span>"
                    "<span style='font-size:12px;color:#b7770d'>Wukᵀ</span>"
                    "<span style='font-size:12px;color:#0057b8'>·cKVᵀ</span>"
                ),
                showarrow=False,
                bgcolor="rgba(255,255,255,0.94)",
                borderpad=3,
            ),
            dict(
                xref="paper", yref="paper", x=0.765, y=0.255,
                text=(
                    "<span style='font-size:13px;font-weight:800;color:#c0392b'>conceptual only in absorbed mode</span>"
                    "<br><span style='font-size:12px;color:#333'>scores skip Q/K; output skips V/C</span>"
                ),
                showarrow=False,
                bgcolor="rgba(255,255,255,0.94)",
                borderpad=3,
            ),
            dict(
                xref="paper", yref="paper", x=0.665, y=0.825,
                text=(
                    "<span style='font-size:15px;font-weight:800;color:#1a5276'>actual score op</span>"
                    "<br><span style='font-size:14px;color:#c0392b'>x_new</span>"
                    "<span style='font-size:14px;color:#1c2833'> @ </span>"
                    "<span style='font-size:14px;color:#b7770d'>(Wq @ Wuk.T)</span>"
                    "<span style='font-size:14px;color:#1c2833'> @ </span>"
                    "<span style='font-size:14px;color:#0057b8'>cKV.T</span>"
                    "<span style='font-size:14px;color:#1c2833'> -> S_new</span>"
                ),
                showarrow=True,
                ax=-40,
                ay=18,
                arrowhead=3,
                arrowwidth=1.6,
                arrowcolor=_rgba("#1a5276", 0.85),
                bgcolor="rgba(255,255,255,0.97)",
                bordercolor=_rgba("#1a5276", 0.45),
                borderwidth=1,
                borderpad=6,
                align="center",
            ),
            dict(
                xref="paper", yref="paper", x=0.710, y=0.165,
                text=(
                    "<span style='font-size:15px;font-weight:800;color:#a04000'>actual output op</span>"
                    "<br><span style='font-size:14px;color:#117a65'>A_new</span>"
                    "<span style='font-size:14px;color:#1c2833'> @ </span>"
                    "<span style='font-size:14px;color:#0057b8'>cKV</span>"
                    "<span style='font-size:14px;color:#1c2833'> @ </span>"
                    "<span style='font-size:14px;color:#b7770d'>(Wuv @ Wo)</span>"
                    "<span style='font-size:14px;color:#1c2833'> -> output_new</span>"
                ),
                showarrow=True,
                ax=-30,
                ay=-22,
                arrowhead=3,
                arrowwidth=1.6,
                arrowcolor=_rgba("#a04000", 0.85),
                bgcolor="rgba(255,255,255,0.97)",
                bordercolor=_rgba("#a04000", 0.45),
                borderwidth=1,
                borderpad=6,
                align="center",
            ),
        ])

    node_list = list(MLA_LAYOUT.keys())
    for name, layout in MLA_LAYOUT.items():
        mat = matrices.get(name)
        if mat is None:
            continue
        is_active = name in active_node_set
        display = mat.copy().astype(float)
        if name == "S_masked":
            display[~np.isfinite(display)] = np.nan
        rows, cols = display.shape
        use_div = name in ("S", "S_scaled", "S_masked")
        is_param_node = layout["nt"] == "param"
        colorscale = "RdBu_r" if use_div else ("YlOrBr" if is_param_node else "Viridis")
        hover = [[f"<b>{name}[{r},{c}]</b><br>{'masked' if np.isnan(display[r, c]) else f'{display[r, c]:.4f}'}"
                  for c in range(cols)] for r in range(rows)]
        idx = node_list.index(name)
        xax = "x" if idx == 0 else f"x{idx+1}"
        yax = "y" if idx == 0 else f"y{idx+1}"
        xak = "xaxis" if idx == 0 else f"xaxis{idx+1}"
        yak = "yaxis" if idx == 0 else f"yaxis{idx+1}"
        x0, y0, x1, y1 = _mla_bounds(name)
        row_labels = [tokens[r] if r < len(tokens) else str(r) for r in range(rows)]
        zmin = zmax = None
        if use_div:
            finite = display[np.isfinite(display)]
            vmax = max(abs(finite.max()), abs(finite.min())) if len(finite) else 1.0
            zmin, zmax = -vmax, vmax
        traces.append((go.Heatmap(
            z=display,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            showscale=False,
            opacity=0.92 if is_active else 0.18,
            hoverinfo="text",
            hovertemplate="%{text}<extra></extra>",
            text=hover,
            y=row_labels if name == "X" else None,
            xaxis=xax,
            yaxis=yax,
        ), xak, yak, x0, y0, x1, y1, name))

    if decode_row_view:
        _blank_old_rows(shapes, _mla_bounds, ("X", "Q", "S", "S_scaled", "S_masked", "A", "C"), seq_len)

    for node_name in ("X", "Q", "cKV", "K", "V", "C"):
        _add_mla_row_overlay(shapes, node_name, seq_len, new_row, new_color, 0.32)
    for node_name in ("S", "S_scaled", "S_masked", "A"):
        _add_mla_row_overlay(shapes, node_name, seq_len, new_row, new_color, 0.42)

    for row in range(seq_len - 1):
        _add_mla_row_overlay(shapes, "cKV", seq_len, row, cached_color, 0.36)

    annotations.extend([
        dict(xref="paper", yref="paper", x=0.330, y=0.420,
             text="<span style='font-size:12px;color:#0057b8'><b>cKV is left operand</b></span><br>"
                  "<span style='font-size:11px;color:#333'>cKV @ Wuk, cKV @ Wuv</span>",
             showarrow=False, xanchor="left", bgcolor="rgba(255,255,255,0.92)",
             borderpad=3, font=dict(size=11, color="#333")),
        dict(xref="paper", yref="paper", x=0.610, y=1.075,
             text="<span style='color:#d35400'><b>orange</b></span> = newest token rows",
             showarrow=False, xanchor="left", bgcolor="rgba(255,255,255,0.92)",
             borderpad=3, font=dict(size=11, color="#333")),
        dict(xref="paper", yref="paper", x=0.610, y=1.035,
             text="<span style='color:#0057b8'><b>blue</b></span> = cached cKV rows only",
             showarrow=False, xanchor="left", bgcolor="rgba(255,255,255,0.92)",
             borderpad=3, font=dict(size=11, color="#333")),
        dict(xref="paper", yref="paper", x=0.610, y=0.995,
             text="K and V are reconstructed from cKV; K/V are not cached",
             showarrow=False, xanchor="left", bgcolor="rgba(255,255,255,0.92)",
             borderpad=3, font=dict(size=11, color="#333")),
        dict(xref="paper", yref="paper", x=0.610, y=0.955,
             text=("Only newest Q row and absorbed output row are used for prediction"
                   if show_absorption else
                   "Only newest Q row and newest C row are used for prediction"),
             showarrow=False, xanchor="left", bgcolor="rgba(255,255,255,0.92)",
             borderpad=3, font=dict(size=11, color="#333")),
    ])
    if decode_row_view:
        annotations.append(dict(xref="paper", yref="paper", x=0.610, y=0.915,
             text="Decode row view: old non-cache rows are hidden",
             showarrow=False, xanchor="left", bgcolor="rgba(255,255,255,0.92)",
             borderpad=3, font=dict(size=11, color="#333")))

    annotations.append(dict(
        xref="paper", yref="paper", x=0.500, y=-0.115,
        text=(
            "<span style='font-size:17px;font-weight:800;color:#1c2833'>Absorption reassociation</span>"
            "<br>"
            "<span style='font-size:15px;color:#1a5276'><b>scores</b></span>"
            "<span style='font-size:15px;color:#1c2833'>: </span>"
            "<span style='font-size:15px;color:#c0392b'>X</span>"
            "<span style='font-size:15px;color:#b7770d'>·Wq</span>"
            "<span style='font-size:15px;color:#0057b8'>·(cKV·Wuk)ᵀ</span>"
            "<span style='font-size:15px;color:#1c2833'> = </span>"
            "<span style='font-size:15px;color:#c0392b'>X</span>"
            "<span style='font-size:15px;color:#b7770d'>·(Wq·Wukᵀ)</span>"
            "<span style='font-size:15px;color:#0057b8'>·cKVᵀ</span>"
            "<br>"
            "<span style='font-size:15px;color:#a04000'><b>output</b></span>"
            "<span style='font-size:15px;color:#1c2833'>: </span>"
            "<span style='font-size:15px;color:#117a65'>A</span>"
            "<span style='font-size:15px;color:#0057b8'>·cKV</span>"
            "<span style='font-size:15px;color:#b7770d'>·Wuv·Wo</span>"
            "<span style='font-size:15px;color:#1c2833'> = </span>"
            "<span style='font-size:15px;color:#117a65'>A</span>"
            "<span style='font-size:15px;color:#0057b8'>·cKV</span>"
            "<span style='font-size:15px;color:#b7770d'>·(Wuv·Wo)</span>"
        ),
        showarrow=False,
        align="center",
        bgcolor="rgba(255,255,255,0.96)",
        bordercolor="rgba(28,40,51,0.18)",
        borderwidth=1,
        borderpad=7,
        font=dict(size=15, color="#1c2833"),
    ))

    if show_absorption:
        annotations.append(dict(
            xref="paper", yref="paper", x=0.500, y=-0.115,
            text=(
                "<span style='font-size:17px;font-weight:800;color:#1c2833'>Absorption trick</span>"
                "<br>"
                "<span style='font-size:15px;color:#1c2833'>Scores: </span>"
                "<span style='font-size:15px;color:#c0392b'>X</span>"
                "<span style='font-size:15px;color:#b7770d'>·Wq</span>"
                "<span style='font-size:15px;color:#0057b8'>·(cKV·Wuk)ᵀ</span>"
                "<span style='font-size:15px;color:#1c2833'> = </span>"
                "<span style='font-size:15px;color:#c0392b'>X</span>"
                "<span style='font-size:15px;color:#b7770d'>·(Wq·Wukᵀ)</span>"
                "<span style='font-size:15px;color:#0057b8'>·cKVᵀ</span>"
                "<br>"
                "<span style='font-size:15px;color:#1c2833'>Output side: </span>"
                "<span style='font-size:15px;color:#117a65'>A</span>"
                "<span style='font-size:15px;color:#0057b8'>·cKV</span>"
                "<span style='font-size:15px;color:#b7770d'>·Wuv·Wo</span>"
                "<span style='font-size:15px;color:#1c2833'> = </span>"
                "<span style='font-size:15px;color:#117a65'>A</span>"
                "<span style='font-size:15px;color:#0057b8'>·cKV</span>"
                "<span style='font-size:15px;color:#b7770d'>·(Wuv·Wo)</span>"
            ),
            showarrow=False,
            align="center",
            bgcolor="rgba(255,255,255,0.96)",
            bordercolor="rgba(28,40,51,0.18)",
            borderwidth=1,
            borderpad=7,
            font=dict(size=15, color="#1c2833"),
        ))

    layout_args = dict(
        paper_bgcolor="white", plot_bgcolor="white",
        width=fig_width, height=fig_height,
        margin=dict(l=8, r=8, t=100, b=130),
        showlegend=False,
        shapes=shapes,
        annotations=annotations,
    )
    for _, xak, yak, x0, y0, x1, y1, name in traces:
        layout_args[xak] = dict(domain=[x0, x1], showticklabels=False,
                                showgrid=False, zeroline=False, fixedrange=True)
        layout_args[yak] = dict(domain=[y0, y1], showticklabels=(name == "X"),
                                tickfont=dict(size=8, color="#444"), side="right",
                                showgrid=False, zeroline=False, fixedrange=True,
                                autorange="reversed")

    return go.Figure(data=[t[0] for t in traces], layout=go.Layout(**layout_args))


def mla_detail_panel(node_name, matrices, tokens):
    mat = matrices.get(node_name)
    if mat is None:
        return go.Figure()
    display = mat.copy().astype(float)
    if node_name == "S_masked":
        display[~np.isfinite(display)] = np.nan
    rows, cols = display.shape
    role, formula = MLA_ROLES.get(node_name, ("", ""))
    title_text = f"<b>{node_name}</b>  <span style='color:#888'>{role}  [{rows}×{cols}]</span>"
    if formula:
        title_text += f"   <span style='color:#aaa'>{formula}</span>"
    row_labels = [tokens[r] if r < len(tokens) else str(r) for r in range(rows)]
    col_labels = [f"d{c}" for c in range(cols)]
    hover = [[("masked" if np.isnan(display[r, c]) else f"{display[r, c]:.4f}")
              for c in range(cols)] for r in range(rows)]
    fig = go.Figure(go.Heatmap(
        z=display,
        colorscale="RdBu_r" if node_name in ("S", "S_scaled", "S_masked") else "Viridis",
        showscale=True,
        hovertemplate="%{text}<extra></extra>",
        text=hover,
        x=col_labels,
        y=row_labels,
    ))
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=12)),
        height=270,
        margin=dict(l=50, r=10, t=48, b=30),
        paper_bgcolor="white",
        plot_bgcolor="white",
        yaxis=dict(autorange="reversed"),
    )
    return fig
