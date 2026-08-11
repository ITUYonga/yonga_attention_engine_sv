#!/usr/bin/env python3
# Generates architecture.drawio (system architecture + per-module FSM pages)
# for the README. Plain mxGraph XML, no external libraries. Re-run after
# editing this script to regenerate; the .drawio file itself is still fully
# editable by hand afterward in the draw.io extension.

import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

_id_counter = [0]


def new_id(prefix="n"):
    _id_counter[0] += 1
    return f"{prefix}{_id_counter[0]}"


OWNER_COLORS = {
    "belinay": "#e1d5e7",  # purple - compute core
    "can":     "#d5e8d4",  # green  - projection
    "hasan":   "#ffe6cc",  # orange - softmax
    "taha":    "#dae8fc",  # blue   - control/memory
    "neutral": "#f5f5f5",
}
OWNER_STROKE = {
    "belinay": "#9673a6",
    "can":     "#82b366",
    "hasan":   "#d79b00",
    "taha":    "#6c8ebf",
    "neutral": "#666666",
}


def block(cells, label, x, y, w, h, owner="neutral", sub=""):
    vid = new_id("b")
    text = label if not sub else f"{label}\n{sub}"
    style = (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=8;fontSize=12;"
        f"fillColor={OWNER_COLORS[owner]};strokeColor={OWNER_STROKE[owner]};"
    )
    cells.append(
        f'<mxCell id="{vid}" value="{esc(text)}" style="{style}" vertex="1" parent="1">'
        f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" /></mxCell>'
    )
    return vid


def group(cells, label, x, y, w, h, owner="neutral"):
    vid = new_id("g")
    style = (
        "rounded=1;whiteSpace=wrap;html=1;verticalAlign=top;fontStyle=1;fontSize=13;"
        f"fillColor=none;strokeColor={OWNER_STROKE[owner]};dashed=1;"
    )
    cells.append(
        f'<mxCell id="{vid}" value="{esc(label)}" style="{style}" vertex="1" parent="1">'
        f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" /></mxCell>'
    )
    return vid


def state(cells, label, x, y, w=140, h=50):
    vid = new_id("s")
    style = "rounded=1;whiteSpace=wrap;html=1;arcSize=40;fontSize=12;fillColor=#f8cecc;strokeColor=#b85450;"
    cells.append(
        f'<mxCell id="{vid}" value="{esc(label)}" style="{style}" vertex="1" parent="1">'
        f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" /></mxCell>'
    )
    return vid


def edge(cells, src, dst, label="", style_extra="", exit_x=None, exit_y=None, entry_x=None, entry_y=None):
    eid = new_id("e")
    style = "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;fontSize=11;" + style_extra
    if exit_x is not None:
        style += f"exitX={exit_x};exitY={exit_y};exitDx=0;exitDy=0;"
    if entry_x is not None:
        style += f"entryX={entry_x};entryY={entry_y};entryDx=0;entryDy=0;"
    cells.append(
        f'<mxCell id="{eid}" value="{esc(label)}" style="{style}" edge="1" parent="1" source="{src}" target="{dst}">'
        f'<mxGeometry relative="1" as="geometry" /></mxCell>'
    )
    return eid


def esc(s):
    return (
        s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace('"', "&quot;").replace("\n", "&#10;")
    )


def page(name, cells, w=1400, h=900):
    pid = new_id("page")
    body = "".join(cells)
    return (
        f'<diagram name="{esc(name)}" id="{pid}">'
        f'<mxGraphModel dx="900" dy="700" grid="1" gridSize="10" guides="1" tooltips="1" '
        f'connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{w}" pageHeight="{h}" math="0" shadow="0">'
        f'<root><mxCell id="0" /><mxCell id="1" parent="0" />{body}</root>'
        f'</mxGraphModel></diagram>'
    )


# ---------------------------------------------------------------------------
# Page 1: System architecture
# ---------------------------------------------------------------------------
def build_architecture():
    cells = []

    axi_in = block(cells, "AXI-Stream In", 40, 260, 140, 60, "taha", "FP32")
    bf16c = block(cells, "bf16_comb", 220, 260, 140, 60, "belinay", "FP32 -> BF16")
    dbuf_ = block(cells, "dbuf", 400, 260, 140, 60, "taha", "ping-pong buffer")
    streamer = block(cells, "dbuf_read_streamer", 580, 260, 160, 60, "taha")

    grp = group(cells, "projection_block", 780, 120, 300, 300, "can")
    proj_q = block(cells, "qkv_proj (Wq)", 800, 160, 260, 40, "can")
    rope_q = block(cells, "rope (Q)", 800, 210, 260, 40, "can")
    proj_k = block(cells, "qkv_proj (Wk)", 800, 260, 260, 40, "can")
    rope_k = block(cells, "rope (K)", 800, 310, 260, 40, "can")
    proj_v = block(cells, "qkv_proj (Wv)  (no RoPE)", 800, 360, 260, 40, "can")
    gqa = block(cells, "gqa_mapper", 800, 410, 260, 30, "can", "head index only")

    uram = block(cells, "URAM ping-pong\n(Q / K / V store)", 1140, 260, 180, 80, "taha")

    modA = block(
        cells, "Module A\nshared SIZE x SIZE PE array", 1140, 400, 220, 90, "belinay",
        "arbiter picks Q.K^T vs score.V by tag\nrouter sends result out by tag",
    )

    scalemask = block(cells, "scale_mask", 1140, 540, 220, 40, "hasan")
    softmax_b = block(cells, "softmax\n(exp_lut / recip_lut / row_fifo)", 1140, 600, 220, 60, "hasan")

    proj_o = block(cells, "qkv_proj (Wo)", 800, 540, 260, 40, "can")
    tx = block(cells, "tx_fifo", 400, 540, 140, 60, "taha")
    bf16v = block(cells, "bf16_convert", 220, 540, 140, 60, "belinay", "BF16 -> FP32")
    axi_out = block(cells, "AXI-Stream Out", 40, 540, 140, 60, "taha", "FP32")

    # main forward flow
    edge(cells, axi_in, bf16c)
    edge(cells, bf16c, dbuf_)
    edge(cells, dbuf_, streamer)
    edge(cells, streamer, proj_q, "token")
    edge(cells, streamer, proj_k, "token")
    edge(cells, streamer, proj_v, "token")
    edge(cells, proj_q, rope_q)
    edge(cells, proj_k, rope_k)

    edge(cells, rope_q, modA, "Q", exit_x="1", exit_y="0.3")
    edge(cells, rope_k, uram, "K")
    edge(cells, proj_v, uram, "V")
    edge(cells, uram, modA, "Q/K read")

    edge(cells, modA, scalemask, "Q.K^T result")
    edge(cells, scalemask, softmax_b)
    edge(cells, softmax_b, modA, "weights (score.V pass)", style_extra="dashed=1;strokeColor=#b85450;", entry_x="1", entry_y="0.7")
    edge(cells, uram, modA, "V read (score.V pass)", style_extra="dashed=1;")

    edge(cells, modA, proj_o, "attention output")
    edge(cells, proj_o, tx)
    edge(cells, tx, bf16v)
    edge(cells, bf16v, axi_out)

    note = block(
        cells,
        "Module A is instantiated ONCE.\nSame physical array computes both\nQ.K^T and score.V, distinguished\nby a 1-bit tag carried with the data.",
        1140, 700, 260, 90, "neutral",
    )

    return page("System Architecture", cells, w=1480, h=830)


# ---------------------------------------------------------------------------
# Page 2: qkv_proj FSM
# ---------------------------------------------------------------------------
def build_qkv_proj_fsm():
    cells = []
    s_load = state(cells, "ST_LOAD\ncollect input vector", 60, 220)
    s_mi = state(cells, "ST_MUL_ISSUE\npulse bf16_mul valid_i", 320, 60)
    s_mw = state(cells, "ST_MUL_WAIT\nwait mul valid_o", 580, 60)
    s_ai = state(cells, "ST_ADD_ISSUE\npulse bf16_add valid_i", 580, 220)
    s_aw = state(cells, "ST_ADD_WAIT\nwait add valid_o", 320, 220)
    s_drain = state(cells, "ST_DRAIN\nstream y_mem out", 60, 400)

    edge(cells, s_load, s_load, "x_valid_i && !x_last_i")
    edge(cells, s_load, s_mi, "x_last_i (vector complete)")
    edge(cells, s_mi, s_mw, "")
    edge(cells, s_mw, s_ai, "mul_valid_o")
    edge(cells, s_ai, s_aw, "")
    edge(cells, s_aw, s_mi, "add_valid_o &&\nmac_idx != D_MODEL-1\n(next weight element)")
    edge(cells, s_aw, s_drain, "add_valid_o && mac_idx==D_MODEL-1\n&& out_idx==D_OUT-1")
    edge(cells, s_aw, s_mi, "add_valid_o && mac_idx==D_MODEL-1\n&& out_idx != D_OUT-1\n(next output element)", style_extra="strokeColor=#82b366;")
    edge(cells, s_drain, s_drain, "y_ready_i && drain_ptr != D_OUT-1")
    edge(cells, s_drain, s_load, "y_ready_i && drain_ptr==D_OUT-1")

    note = block(
        cells,
        "One shared multiplier + one shared adder,\nreused for every MAC step of every output\nelement. y = W*x, D_OUT*D_MODEL MAC steps total.",
        850, 220, 280, 90, "can",
    )
    return page("FSM - qkv_proj", cells, w=1200, h=560)


# ---------------------------------------------------------------------------
# Page 3: rope FSM
# ---------------------------------------------------------------------------
def build_rope_fsm():
    cells = []
    s_load = state(cells, "ST_LOAD\ncollect vector + latch pos_i", 60, 220)
    s_mi = state(cells, "ST_MUL_ISSUE\nmstep 0..3", 320, 60)
    s_mw = state(cells, "ST_MUL_WAIT", 580, 60)
    s_ai = state(cells, "ST_ADD_ISSUE\n(mstep 1 or 3 only)", 580, 220)
    s_aw = state(cells, "ST_ADD_WAIT", 320, 220)
    s_drain = state(cells, "ST_DRAIN\nstream rotated vector out", 60, 400)

    edge(cells, s_load, s_load, "x_valid_i && !x_last_i")
    edge(cells, s_load, s_mi, "x_last_i")
    edge(cells, s_mi, s_mw, "")
    edge(cells, s_mw, s_mi, "mul_valid_o && mstep in {0,2}\n(x1*cos or x1*sin, no add needed)", style_extra="strokeColor=#82b366;")
    edge(cells, s_mw, s_ai, "mul_valid_o && mstep in {1,3}\n(needs add/sub next)")
    edge(cells, s_ai, s_aw, "")
    edge(cells, s_aw, s_mi, "add_valid_o && mstep==1\n(y1 done, start mstep 2)")
    edge(cells, s_aw, s_mi, "add_valid_o && mstep==3 &&\npair_idx != PAIR_CNT-1\n(y2 done, next pair)", style_extra="strokeColor=#82b366;")
    edge(cells, s_aw, s_drain, "add_valid_o && mstep==3 &&\npair_idx==PAIR_CNT-1\n(last pair done)")
    edge(cells, s_drain, s_drain, "y_ready_i && drain_ptr != D_MODEL-1")
    edge(cells, s_drain, s_load, "y_ready_i && drain_ptr==D_MODEL-1")

    note = block(
        cells,
        "4 micro-steps per pair (mstep 0..3):\n0: x1*cos -> tmp1\n1: x2*sin, then tmp1-result = y1\n2: x1*sin -> tmp3\n3: x2*cos, then tmp3+result = y2\nPairing is rotate-half: x1=x[i], x2=x[i+D_MODEL/2]",
        850, 200, 300, 140, "can",
    )
    return page("FSM - rope", cells, w=1220, h=560)


# ---------------------------------------------------------------------------
# Page 4: softmax FSM
# ---------------------------------------------------------------------------
def build_softmax_fsm():
    cells = []
    s_idle = state(cells, "IDLE", 60, 220, w=160)
    s_acc = state(cells, "ACCUMULATE\nexp_lut -> bf16_add -> sum_acc", 340, 220, w=220)
    s_inv = state(cells, "INVERT\nrecip_lut -> inv_sum_reg", 660, 220, w=200)
    s_div = state(cells, "DIVIDE_NORMALIZE\nfifo_dout * inv_sum_reg -> output", 960, 220, w=260)

    edge(cells, s_idle, s_acc, "s_axis_tvalid && s_axis_tready")
    edge(cells, s_acc, s_acc, "more row elements arriving")
    edge(cells, s_acc, s_inv, "add_valid_o && last_d2\n(last element's sum landed)")
    edge(cells, s_inv, s_div, "(1 cycle, latch reciprocal)")
    edge(cells, s_div, s_div, "fifo not empty yet")
    edge(cells, s_div, s_idle, "fifo_empty && m_axis_tvalid\n&& m_axis_tready", style_extra="strokeColor=#82b366;")

    note = block(
        cells,
        "Row-wise softmax, 2 passes over the row:\nPass 1 (ACCUMULATE): exp(x_i) into row_fifo, sum into sum_acc\nPass 2 (DIVIDE_NORMALIZE): read row_fifo back, multiply by 1/sum",
        340, 380, 420, 90, "hasan",
    )
    return page("FSM - softmax", cells, w=1300, h=520)


# ---------------------------------------------------------------------------
# Page 5: uram_pingpong_controller read FSM
# ---------------------------------------------------------------------------
def build_uram_fsm():
    cells = []
    s_idle = state(cells, "ST_IDLE", 60, 220, w=160)
    s_qk = state(cells, "ST_READ_QK\nq_ren=1, k_ren=1", 340, 220, w=200)
    s_wait = state(cells, "ST_WAIT_SOFTMAX", 660, 220, w=200)
    s_v = state(cells, "ST_READ_V\nv_ren=1", 960, 220, w=200)

    edge(cells, s_idle, s_qk, "pending_packages > 0")
    edge(cells, s_qk, s_qk, "rd_ptr != MATRIX_SIZE-1")
    edge(cells, s_qk, s_wait, "rd_ptr==MATRIX_SIZE-1")
    edge(cells, s_wait, s_v, "softmax_valid")
    edge(cells, s_v, s_v, "rd_ptr != MATRIX_SIZE-1")
    edge(cells, s_v, s_idle, "rd_ptr==MATRIX_SIZE-1\n(swap rd_bank)", style_extra="strokeColor=#82b366;")

    note = block(
        cells,
        "Write side (not shown) is a separate always_ff:\nQ/K/V land in the write-side ping-pong bank\nwhile Module A reads the other bank via this FSM.",
        340, 380, 420, 80, "taha",
    )
    return page("FSM - uram_pingpong_controller", cells, w=1240, h=520)


def main():
    pages = [
        build_architecture(),
        build_qkv_proj_fsm(),
        build_rope_fsm(),
        build_softmax_fsm(),
        build_uram_fsm(),
    ]
    xml_str = '<mxfile host="app.diagrams.net">' + "".join(pages) + "</mxfile>"

    # pretty-print for a readable diff / manual editing later
    pretty = minidom.parseString(xml_str).toprettyxml(indent="  ")
    pretty = "\n".join(line for line in pretty.split("\n") if line.strip())

    out_path = "architecture.drawio"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(pretty)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
