#!/usr/bin/env python3
# Generates architecture.drawio (system architecture + per-module FSM pages)
# AND matching PNGs under images/, from the same box/edge data so the two
# never drift apart. Style: white background, black ink, Courier New —
# a plain 1970s-technical-paper block-diagram look, no color fills.
#
# Re-run after editing this script (or after an RTL state machine changes)
# to regenerate both outputs:
#   python scripts/gen_diagrams.py

import os
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Ellipse
from matplotlib.path import Path

FONT = "Courier New"
plt.rcParams["font.family"] = FONT

_id_counter = [0]


def new_id(prefix="n"):
    _id_counter[0] += 1
    return f"{prefix}{_id_counter[0]}"


# ---------------------------------------------------------------------------
# Shared data model: a page is (name, boxes, edges, page_w, page_h).
# box:  {id, label, x, y, w, h, shape}   shape in {"rect", "ellipse"}
# edge: {src, dst, label, dashed, loop}
# ---------------------------------------------------------------------------

def box(boxes, label, x, y, w, h, shape="rect"):
    bid = new_id("b")
    boxes.append({"id": bid, "label": label, "x": x, "y": y, "w": w, "h": h, "shape": shape})
    return bid


def group(boxes, label, x, y, w, h):
    return box(boxes, label, x, y, w, h, shape="group")


def edge(edges, src, dst, label="", dashed=False, rad=None, loop_side="top"):
    edges.append({
        "src": src, "dst": dst, "label": label, "dashed": dashed,
        "loop": src == dst, "force_rad": rad, "loop_side": loop_side,
    })


# ---------------------------------------------------------------------------
# Page 1: System architecture
# ---------------------------------------------------------------------------
def build_architecture():
    boxes, edges = [], []

    axi_in = box(boxes, "AXI-Stream In\n(FP32)", 40, 260, 150, 60)
    bf16c = box(boxes, "bf16_comb\nFP32 -> BF16", 230, 260, 150, 60)
    dbuf_ = box(boxes, "dbuf\nping-pong buffer", 420, 260, 150, 60)
    streamer = box(boxes, "dbuf_read_streamer", 610, 260, 170, 60)

    grp = group(boxes, "projection_block", 810, 100, 320, 370)
    proj_q = box(boxes, "qkv_proj (Wq)", 830, 150, 280, 45)
    rope_q = box(boxes, "rope (Q)", 830, 205, 280, 45)
    proj_k = box(boxes, "qkv_proj (Wk)", 830, 260, 280, 45)
    rope_k = box(boxes, "rope (K)", 830, 315, 280, 45)
    proj_v = box(boxes, "qkv_proj (Wv) -- no RoPE", 830, 370, 280, 45)
    gqa = box(boxes, "gqa_mapper -- head index only", 830, 425, 280, 35)

    uram = box(boxes, "URAM ping-pong\n(Q / K / V store)", 1260, 250, 200, 90)

    modA = box(
        boxes, "Module A\nshared SIZE x SIZE PE array\n\n"
        "arbiter: Q.K^T vs score.V by tag\nrouter: sends result out by tag",
        1260, 400, 250, 140,
    )

    scalemask = box(boxes, "scale_mask", 1600, 400, 200, 45)
    softmax_b = box(boxes, "softmax\n(exp_lut / recip_lut / row_fifo)", 1600, 470, 200, 70)

    proj_o = box(boxes, "qkv_proj (Wo)", 830, 620, 280, 45)
    tx = box(boxes, "tx_fifo", 420, 620, 150, 60)
    bf16v = box(boxes, "bf16_convert\nBF16 -> FP32", 230, 620, 150, 60)
    axi_out = box(boxes, "AXI-Stream Out\n(FP32)", 40, 620, 150, 60)

    edge(edges, axi_in, bf16c)
    edge(edges, bf16c, dbuf_)
    edge(edges, dbuf_, streamer)
    edge(edges, streamer, proj_q, "token")
    edge(edges, streamer, proj_k, "token")
    edge(edges, streamer, proj_v, "token")
    edge(edges, proj_q, rope_q)
    edge(edges, proj_k, rope_k)

    edge(edges, rope_q, modA, "Q")
    edge(edges, rope_k, uram, "K")
    edge(edges, proj_v, uram, "V")
    edge(edges, uram, modA, "Q/K read")

    edge(edges, modA, scalemask, "Q.K^T result")
    edge(edges, scalemask, softmax_b)
    edge(edges, softmax_b, modA, "weights\n(score.V pass)", dashed=True)
    edge(edges, uram, modA, "V read\n(score.V pass)", dashed=True)

    edge(edges, modA, proj_o, "attention output")
    edge(edges, proj_o, tx)
    edge(edges, tx, bf16v)
    edge(edges, bf16v, axi_out)

    note = box(
        boxes,
        "NOTE: Module A is instantiated ONCE.\n"
        "The same physical array computes both\n"
        "Q.K^T and score.V, distinguished by a\n"
        "1-bit tag carried alongside the data.",
        1260, 620, 300, 100,
    )

    return ("System Architecture", boxes, edges, 1870, 760)


# ---------------------------------------------------------------------------
# Page 2: qkv_proj FSM
# ---------------------------------------------------------------------------
def build_qkv_proj_fsm():
    boxes, edges = [], []
    s_load = box(boxes, "ST_LOAD\ncollect input vector", 60, 220, 200, 60, "ellipse")
    s_mi = box(boxes, "ST_MUL_ISSUE\npulse bf16_mul valid_i", 380, 40, 220, 60, "ellipse")
    s_mw = box(boxes, "ST_MUL_WAIT\nwait mul valid_o", 700, 40, 200, 60, "ellipse")
    s_ai = box(boxes, "ST_ADD_ISSUE\npulse bf16_add valid_i", 700, 220, 220, 60, "ellipse")
    s_aw = box(boxes, "ST_ADD_WAIT\nwait add valid_o", 380, 220, 200, 60, "ellipse")
    s_drain = box(boxes, "ST_DRAIN\nstream y_mem out", 60, 400, 200, 60, "ellipse")

    edge(edges, s_load, s_load, "x_valid_i && !x_last_i")
    edge(edges, s_load, s_mi, "x_last_i (vector complete)")
    edge(edges, s_mi, s_mw, "")
    edge(edges, s_mw, s_ai, "mul_valid_o")
    edge(edges, s_ai, s_aw, "")
    edge(edges, s_aw, s_mi, "add_valid_o, mac_idx!=D_MODEL-1\n(next weight element)")
    edge(edges, s_aw, s_drain, "add_valid_o, mac_idx==D_MODEL-1\n&& out_idx==D_OUT-1")
    edge(edges, s_aw, s_mi, "add_valid_o, mac_idx==D_MODEL-1\n&& out_idx!=D_OUT-1\n(next output element)")
    edge(edges, s_drain, s_drain, "y_ready_i && drain_ptr!=D_OUT-1", loop_side="bottom")
    edge(edges, s_drain, s_load, "y_ready_i && drain_ptr==D_OUT-1")

    note = box(
        boxes,
        "One shared multiplier + one shared adder,\n"
        "reused for every MAC step of every output\n"
        "element. y = W*x, D_OUT*D_MODEL MAC steps total.",
        60, 580, 860, 70,
    )
    return ("FSM - qkv_proj", boxes, edges, 1000, 700)


# ---------------------------------------------------------------------------
# Page 3: rope FSM
# ---------------------------------------------------------------------------
def build_rope_fsm():
    boxes, edges = [], []
    s_load = box(boxes, "ST_LOAD\ncollect vector + latch pos_i", 60, 220, 220, 60, "ellipse")
    s_mi = box(boxes, "ST_MUL_ISSUE\nmstep 0..3", 380, 40, 200, 60, "ellipse")
    s_mw = box(boxes, "ST_MUL_WAIT", 700, 40, 180, 60, "ellipse")
    s_ai = box(boxes, "ST_ADD_ISSUE\n(mstep 1 or 3 only)", 700, 220, 220, 60, "ellipse")
    s_aw = box(boxes, "ST_ADD_WAIT", 380, 220, 180, 60, "ellipse")
    s_drain = box(boxes, "ST_DRAIN\nstream rotated vector out", 60, 400, 220, 60, "ellipse")

    edge(edges, s_load, s_load, "x_valid_i && !x_last_i")
    edge(edges, s_load, s_mi, "x_last_i")
    edge(edges, s_mi, s_mw, "")
    edge(edges, s_mw, s_mi, "mul_valid_o, mstep in 0,2\n(x1*cos or x1*sin, no add)")
    edge(edges, s_mw, s_ai, "mul_valid_o, mstep in 1,3\n(needs add/sub next)")
    edge(edges, s_ai, s_aw, "")
    edge(edges, s_aw, s_mi, "add_valid_o, mstep==1\n(y1 done, start mstep 2)")
    edge(edges, s_aw, s_mi, "add_valid_o, mstep==3,\npair_idx!=PAIR_CNT-1\n(y2 done, next pair)")
    edge(edges, s_aw, s_drain, "add_valid_o, mstep==3,\npair_idx==PAIR_CNT-1\n(last pair done)")
    edge(edges, s_drain, s_drain, "y_ready_i && drain_ptr!=D_MODEL-1", loop_side="bottom")
    edge(edges, s_drain, s_load, "y_ready_i && drain_ptr==D_MODEL-1")

    note = box(
        boxes,
        "4 micro-steps per pair (mstep 0..3):\n"
        "0: x1*cos -> tmp1\n"
        "1: x2*sin, then tmp1-result = y1\n"
        "2: x1*sin -> tmp3\n"
        "3: x2*cos, then tmp3+result = y2\n"
        "Pairing is rotate-half: x1=x[i], x2=x[i+D_MODEL/2]",
        60, 580, 860, 130,
    )
    return ("FSM - rope", boxes, edges, 1000, 750)


# ---------------------------------------------------------------------------
# Page 4: softmax FSM
# ---------------------------------------------------------------------------
def build_softmax_fsm():
    boxes, edges = [], []
    s_idle = box(boxes, "IDLE", 60, 240, 180, 80, "ellipse")
    s_acc = box(boxes, "ACCUMULATE\nexp_lut -> bf16_add -> sum_acc", 480, 240, 380, 80, "ellipse")
    s_inv = box(boxes, "INVERT\nrecip_lut -> inv_sum_reg", 1100, 240, 320, 80, "ellipse")
    s_div = box(boxes, "DIVIDE_NORMALIZE\nfifo_dout * inv_sum_reg -> output", 1660, 240, 420, 80, "ellipse")

    edge(edges, s_idle, s_acc, "s_axis_tvalid && s_axis_tready")
    edge(edges, s_acc, s_acc, "more row elements arriving")
    edge(edges, s_acc, s_inv, "add_valid_o && last_d2\n(last element's sum landed)")
    edge(edges, s_inv, s_div, "(1 cycle, latch reciprocal)")
    edge(edges, s_div, s_div, "fifo not empty yet")
    edge(edges, s_div, s_idle, "fifo_empty && m_axis_tvalid && m_axis_tready", rad=-0.35)

    note = box(
        boxes,
        "Row-wise softmax, 2 passes over the row:\n"
        "Pass 1 (ACCUMULATE): exp(x_i) into row_fifo, sum into sum_acc\n"
        "Pass 2 (DIVIDE_NORMALIZE): read row_fifo back, multiply by 1/sum",
        60, 470, 700, 100,
    )
    return ("FSM - softmax", boxes, edges, 2160, 620)


# ---------------------------------------------------------------------------
# Page 5: uram_pingpong_controller read FSM
# ---------------------------------------------------------------------------
def build_uram_fsm():
    boxes, edges = [], []
    s_idle = box(boxes, "ST_IDLE", 60, 220, 160, 60, "ellipse")
    s_qk = box(boxes, "ST_READ_QK\nq_ren=1, k_ren=1", 340, 220, 220, 60, "ellipse")
    s_wait = box(boxes, "ST_WAIT_SOFTMAX", 680, 220, 220, 60, "ellipse")
    s_v = box(boxes, "ST_READ_V\nv_ren=1", 1000, 220, 200, 60, "ellipse")

    edge(edges, s_idle, s_qk, "pending_packages > 0")
    edge(edges, s_qk, s_qk, "rd_ptr != MATRIX_SIZE-1")
    edge(edges, s_qk, s_wait, "rd_ptr==MATRIX_SIZE-1")
    edge(edges, s_wait, s_v, "softmax_valid")
    edge(edges, s_v, s_v, "rd_ptr != MATRIX_SIZE-1")
    edge(edges, s_v, s_idle, "rd_ptr==MATRIX_SIZE-1 (swap rd_bank)", rad=-0.35)

    note = box(
        boxes,
        "Write side (not shown) is a separate always_ff:\n"
        "Q/K/V land in the write-side ping-pong bank\n"
        "while Module A reads the other bank via this FSM.",
        60, 460, 460, 80,
    )
    return ("FSM - uram_pingpong_controller", boxes, edges, 1280, 530)


# ---------------------------------------------------------------------------
# drawio (mxGraph XML) rendering -- monochrome, Courier New
# ---------------------------------------------------------------------------
def box_style(shape):
    common = f"whiteSpace=wrap;html=1;fontFamily={FONT};fontColor=#000000;fillColor=#FFFFFF;strokeColor=#000000;fontSize=12;"
    if shape == "ellipse":
        return "ellipse;" + common
    if shape == "group":
        return f"rounded=0;whiteSpace=wrap;html=1;fontFamily={FONT};fontColor=#000000;fillColor=none;strokeColor=#000000;dashed=1;verticalAlign=top;fontSize=12;fontStyle=1;"
    return "rounded=0;" + common


def render_drawio_page(name, boxes, edges, w, h):
    cells = []
    for b in boxes:
        cells.append(
            f'<mxCell id="{b["id"]}" value="{esc(b["label"])}" style="{box_style(b["shape"])}" '
            f'vertex="1" parent="1"><mxGeometry x="{b["x"]}" y="{b["y"]}" width="{b["w"]}" '
            f'height="{b["h"]}" as="geometry" /></mxCell>'
        )
    pair_seen = {}
    for e in edges:
        if e["loop"]:
            continue
        key = (e["src"], e["dst"])
        pair_seen[key] = pair_seen.get(key, 0) + 1

    pair_idx = {}
    for e in edges:
        eid = new_id("e")
        style = f"edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;fontFamily={FONT};fontColor=#000000;strokeColor=#000000;fontSize=10;"
        if e["dashed"]:
            style += "dashed=1;"
        if e["loop"]:
            style += "curved=1;exitX=0.25;exitY=0;entryX=0.75;entryY=0;"
        else:
            key = (e["src"], e["dst"])
            if pair_seen.get(key, 1) > 1:
                idx = pair_idx.get(key, 0)
                pair_idx[key] = idx + 1
                style += f"curved=1;exitDx={idx * 20};entryDx={idx * 20};"
        cells.append(
            f'<mxCell id="{eid}" value="{esc(e["label"])}" style="{style}" edge="1" parent="1" '
            f'source="{e["src"]}" target="{e["dst"]}"><mxGeometry relative="1" as="geometry" /></mxCell>'
        )
    pid = new_id("page")
    body = "".join(cells)
    return (
        f'<diagram name="{esc(name)}" id="{pid}">'
        f'<mxGraphModel dx="900" dy="700" grid="1" gridSize="10" guides="1" tooltips="1" '
        f'connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{w}" pageHeight="{h}" '
        f'background="#FFFFFF" math="0" shadow="0">'
        f'<root><mxCell id="0" /><mxCell id="1" parent="0" />{body}</root>'
        f'</mxGraphModel></diagram>'
    )


def esc(s):
    return (
        s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace('"', "&quot;").replace("\n", "&#10;")
    )


# ---------------------------------------------------------------------------
# Generic .drawio (mxGraph XML) reader -- used to pick up hand-edits made in
# the draw.io / diagrams.net extension after this script generated a page,
# so re-running the script never clobbers manual layout tweaks. Only the
# flat parent="1" cell structure this script itself produces is supported.
# ---------------------------------------------------------------------------
def parse_style(style_str):
    d = {}
    for part in (style_str or "").split(";"):
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            d[k] = v
        else:
            d[part] = "1"
    return d


def read_existing_pages(path):
    """Returns {page_name: (raw_diagram_xml_str, boxes, edges, w, h)} or {} if
    the file doesn't exist / can't be parsed."""
    if not os.path.exists(path):
        return {}
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return {}
    pages = {}
    for diagram in tree.getroot().findall("diagram"):
        name = diagram.get("name", "page")
        gm = diagram.find("mxGraphModel")
        if gm is None:
            continue
        w = float(gm.get("pageWidth", 1000))
        h = float(gm.get("pageHeight", 800))
        root_el = gm.find("root")
        boxes, edges = [], []
        vertex_ids = set()
        for cell in root_el.findall("mxCell"):
            if cell.get("vertex") == "1":
                geom = cell.find("mxGeometry")
                if geom is None:
                    continue
                style = parse_style(cell.get("style"))
                shape = "ellipse" if "ellipse" in style else ("group" if style.get("verticalAlign") == "top" else "rect")
                boxes.append({
                    "id": cell.get("id"),
                    "label": cell.get("value") or "",
                    "x": float(geom.get("x", 0)), "y": float(geom.get("y", 0)),
                    "w": float(geom.get("width", 100)), "h": float(geom.get("height", 40)),
                    "shape": shape,
                    "fill": style.get("fillColor", "#FFFFFF"),
                    "stroke": style.get("strokeColor", "#000000"),
                })
                vertex_ids.add(cell.get("id"))
        for cell in root_el.findall("mxCell"):
            if cell.get("edge") == "1":
                style = parse_style(cell.get("style"))
                src, dst = cell.get("source"), cell.get("target")
                if src not in vertex_ids or dst not in vertex_ids:
                    continue
                edges.append({
                    "src": src, "dst": dst, "label": cell.get("value") or "",
                    "dashed": style.get("dashed") == "1", "loop": src == dst,
                    "force_rad": None, "loop_side": "top",
                })
        raw_xml = ET.tostring(diagram, encoding="unicode")
        pages[name] = (raw_xml, boxes, edges, w, h)
    return pages


# ---------------------------------------------------------------------------
# PNG rendering via matplotlib -- same monochrome/Courier look
# ---------------------------------------------------------------------------
def clip_to_box(cx, cy, hw, hh, tx, ty):
    # Clip the ray from box center (cx,cy) toward (tx,ty) to the box's edge.
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return cx, cy
    scale = min(
        (hw / abs(dx)) if dx != 0 else float("inf"),
        (hh / abs(dy)) if dy != 0 else float("inf"),
    )
    return cx + dx * scale, cy + dy * scale


def orthogonal_route(scx, scy, shw, shh, dcx, dcy, dhw, dhh, offset=0):
    # Right-angle (Manhattan-style) connector, matching drawio's default
    # edgeStyle=orthogonalEdgeStyle look, instead of a diagonal straight line.
    # Exits/enters through whichever side faces the other box; a Z-shaped
    # two-bend path with the bend line nudged by `offset` keeps parallel
    # edges between the same pair of boxes visually separate.
    dx, dy = dcx - scx, dcy - scy
    if abs(dx) >= abs(dy):
        p1 = (scx + (shw if dx >= 0 else -shw), scy)
        p2 = (dcx - (dhw if dx >= 0 else -dhw), dcy)
        mid_x = (p1[0] + p2[0]) / 2 + offset
        pts = [p1, (mid_x, p1[1]), (mid_x, p2[1]), p2]
    else:
        p1 = (scx, scy + (shh if dy >= 0 else -shh))
        p2 = (dcx, dcy - (dhh if dy >= 0 else -dhh))
        mid_y = (p1[1] + p2[1]) / 2 + offset
        pts = [p1, (p1[0], mid_y), (p2[0], mid_y), p2]
    return pts


def render_png_page(name, boxes, edges, w, h, out_path):
    fig_w, fig_h = w / 100.0, h / 100.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.invert_yaxis()
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_title(name, fontsize=16, family=FONT, color="black", pad=14)

    centers = {}
    for b in boxes:
        cx, cy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
        centers[b["id"]] = (cx, cy, b["w"] / 2, b["h"] / 2)
        fill = b.get("fill", "#FFFFFF")
        fill = "none" if fill in ("none", None) else fill
        stroke = b.get("stroke", "#000000")
        if b["shape"] == "group":
            patch = FancyBboxPatch(
                (b["x"], b["y"]), b["w"], b["h"],
                boxstyle="square,pad=0", facecolor="none", edgecolor=stroke,
                linewidth=1.1, linestyle="dashed", zorder=1,
            )
            ax.add_patch(patch)
            ax.text(
                b["x"] + 10, b["y"] + 8, b["label"], ha="left", va="top", fontsize=9.5,
                family=FONT, color="black", zorder=1, fontweight="bold",
            )
            continue
        if b["shape"] == "ellipse":
            patch = Ellipse((cx, cy), b["w"], b["h"], facecolor=fill, edgecolor=stroke, linewidth=1.3, zorder=2)
        else:
            patch = FancyBboxPatch(
                (b["x"], b["y"]), b["w"], b["h"],
                boxstyle="square,pad=0", facecolor=fill, edgecolor=stroke, linewidth=1.3, zorder=2,
            )
        ax.add_patch(patch)
        ax.text(
            cx, cy, b["label"], ha="center", va="center", fontsize=8.5,
            family=FONT, color="black", zorder=3, linespacing=1.4,
        )

    # Parallel edges (same src/dst pair, e.g. two different guard conditions
    # both going ST_ADD_WAIT -> ST_MUL_ISSUE) would otherwise draw as one
    # straight line with overlapping labels -- bow each successive one out.
    pair_seen = {}
    for e in edges:
        if e["loop"]:
            continue
        key = (e["src"], e["dst"])
        idx = pair_seen.get(key, 0)
        pair_seen[key] = idx + 1
        auto_rad = 0.0 if idx == 0 else 0.28 * idx * (1 if idx % 2 else -1)
        e["_rad"] = e["force_rad"] if e["force_rad"] is not None else auto_rad

    for e in edges:
        scx, scy, shw, shh = centers[e["src"]]
        dcx, dcy, dhw, dhh = centers[e["dst"]]
        if e["loop"]:
            side = e.get("loop_side", "top")
            if side == "top":
                p1, p2 = (scx - shw * 0.18, scy - shh), (scx + shw * 0.18, scy - shh)
                conn, label_pos = "arc3,rad=-1.6", (scx, scy - shh - 55)
            elif side == "bottom":
                p1, p2 = (scx - shw * 0.18, scy + shh), (scx + shw * 0.18, scy + shh)
                conn, label_pos = "arc3,rad=1.6", (scx, scy + shh + 55)
            elif side == "right":
                p1, p2 = (scx + shw, scy - shh * 0.18), (scx + shw, scy + shh * 0.18)
                conn, label_pos = "arc3,rad=-1.6", (scx + shw + 95, scy)
            else:  # left
                p1, p2 = (scx - shw, scy - shh * 0.18), (scx - shw, scy + shh * 0.18)
                conn, label_pos = "arc3,rad=1.6", (scx - shw - 95, scy)
            arrow = FancyArrowPatch(
                p1, p2, connectionstyle=conn, arrowstyle="-|>",
                mutation_scale=12, linewidth=1.0, color="black", zorder=1,
                linestyle="dashed" if e["dashed"] else "solid",
            )
            ax.add_patch(arrow)
            ax.text(*label_pos, e["label"], ha="center", va="center",
                     fontsize=7, family=FONT, color="black")
        elif e["force_rad"] is not None:
            # Explicit long "return" edge (e.g. last state back to the first)
            # -- route it below every box on the row instead of letting the
            # generic router cut straight through same-height states.
            drop = 100
            p1 = (scx, scy + shh)
            p2 = (dcx, dcy + dhh)
            bend_y = max(p1[1], p2[1]) + drop
            pts = [p1, (p1[0], bend_y), (p2[0], bend_y), p2]
            path = Path(pts, [Path.MOVETO] + [Path.LINETO] * (len(pts) - 1))
            arrow = FancyArrowPatch(
                path=path, arrowstyle="-|>", mutation_scale=12, linewidth=1.0,
                color="black", zorder=1, linestyle="dashed" if e["dashed"] else "solid",
            )
            ax.add_patch(arrow)
            if e["label"]:
                # label sits on the middle (bend) segment of the path
                mx = (pts[1][0] + pts[2][0]) / 2
                my = (pts[1][1] + pts[2][1]) / 2
                ax.text(mx, my, e["label"], ha="center", va="center", fontsize=7,
                         family=FONT, color="black",
                         bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
        else:
            # e["_rad"] doubles as a bend-line offset (in page units) here,
            # so parallel edges between the same two boxes still separate.
            offset = (e["_rad"] or 0.0) * 120
            pts = orthogonal_route(scx, scy, shw, shh, dcx, dcy, dhw, dhh, offset=offset)
            path = Path(pts, [Path.MOVETO] + [Path.LINETO] * (len(pts) - 1))
            arrow = FancyArrowPatch(
                path=path, arrowstyle="-|>", mutation_scale=12, linewidth=1.0,
                color="black", zorder=1, linestyle="dashed" if e["dashed"] else "solid",
            )
            ax.add_patch(arrow)
            if e["label"]:
                mx = (pts[1][0] + pts[2][0]) / 2
                my = (pts[1][1] + pts[2][1]) / 2
                ax.text(mx, my, e["label"], ha="center", va="center", fontsize=7,
                         family=FONT, color="black",
                         bbox=dict(facecolor="white", edgecolor="none", pad=0.5))

    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


DRAWIO_PATH = "architecture.drawio"


def main():
    # If architecture.drawio already exists (possibly hand-edited in the
    # draw.io extension since the last run), preserve the "System
    # Architecture" page byte-for-byte instead of regenerating it -- only
    # the FSM pages get freshly rebuilt from the RTL-derived data below.
    existing = read_existing_pages(DRAWIO_PATH)

    fsm_pages = [
        build_qkv_proj_fsm(),
        build_rope_fsm(),
        build_softmax_fsm(),
        build_uram_fsm(),
    ]
    fsm_slugs = ["fsm_qkv_proj", "fsm_rope", "fsm_softmax", "fsm_uram_pingpong"]

    arch_name = "System Architecture"
    if arch_name in existing:
        arch_raw_xml, arch_boxes, arch_edges, arch_w, arch_h = existing[arch_name]
        print(f"kept hand-edited '{arch_name}' page from {DRAWIO_PATH} as-is")
    else:
        arch_page = build_architecture()
        arch_raw_xml = render_drawio_page(*arch_page)
        _, arch_boxes, arch_edges, arch_w, arch_h = arch_page

    fsm_xml = [render_drawio_page(*p) for p in fsm_pages]
    xml_str = '<mxfile host="app.diagrams.net">' + arch_raw_xml + "".join(fsm_xml) + "</mxfile>"
    pretty = minidom.parseString(xml_str).toprettyxml(indent="  ")
    pretty = "\n".join(line for line in pretty.split("\n") if line.strip())
    with open(DRAWIO_PATH, "w", encoding="utf-8") as f:
        f.write(pretty)
    print(f"wrote {DRAWIO_PATH}")

    os.makedirs("images", exist_ok=True)
    render_png_page(arch_name, arch_boxes, arch_edges, arch_w, arch_h, "images/diagram_architecture.png")
    print("wrote images/diagram_architecture.png")
    for (name, boxes, edges, w, h), slug in zip(fsm_pages, fsm_slugs):
        out = f"images/diagram_{slug}.png"
        render_png_page(name, boxes, edges, w, h, out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
