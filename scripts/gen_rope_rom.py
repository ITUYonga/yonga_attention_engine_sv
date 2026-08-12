#!/usr/bin/env python3
# Generates rope_sin_rom.mem / rope_cos_rom.mem for rtl/projection/rope.sv.
#
# Addressing must match rope.sv exactly:
#   rom_addr = pos * PAIR_CNT + pair_idx,  PAIR_CNT = D_MODEL/2
#   freq_i   = base ** (-2*i / D_MODEL),   i = pair_idx (standard RoPE, base=10000)
#   theta    = pos * freq_i
#
# Each line is one bf16 value in plain 4-digit hex (as $readmemh expects).

import argparse
import struct


def to_bf16_hex(x: float) -> str:
    packed = struct.pack(">f", x)
    top16 = packed[0:2]
    return top16.hex()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--max-pos", type=int, default=512)
    ap.add_argument("--base", type=float, default=10000.0)
    ap.add_argument("--out-dir", default="rtl/projection")
    args = ap.parse_args()

    pair_cnt = args.d_model // 2
    sin_lines = []
    cos_lines = []

    for pos in range(args.max_pos):
        for i in range(pair_cnt):
            freq = args.base ** (-2.0 * i / args.d_model)
            theta = pos * freq
            import math
            sin_lines.append(to_bf16_hex(math.sin(theta)))
            cos_lines.append(to_bf16_hex(math.cos(theta)))

    sin_path = f"{args.out_dir}/rope_sin_rom.mem"
    cos_path = f"{args.out_dir}/rope_cos_rom.mem"
    with open(sin_path, "w") as f:
        f.write("\n".join(sin_lines) + "\n")
    with open(cos_path, "w") as f:
        f.write("\n".join(cos_lines) + "\n")

    print(f"wrote {len(sin_lines)} entries to {sin_path}")
    print(f"wrote {len(cos_lines)} entries to {cos_path}")


if __name__ == "__main__":
    main()
