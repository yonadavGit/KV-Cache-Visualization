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
