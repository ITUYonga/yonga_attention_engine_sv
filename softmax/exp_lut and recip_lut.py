import numpy as np
import struct
import os

print("Files are being written to:", os.getcwd())

"""Converts a python flaot into a 4-digit BF16 hex string. (such as: 16'h3F80 => 3F80)"""
def float_to_bf16_hex(val: float) -> str:
    f32_bytes = struct.pack('>f', float(val))
    f32_int   = struct.unpack('>I', f32_bytes)[0]

    bf16_int = (f32_int >> 16) & 0xFFFF
    return f"{bf16_int:04X}"


"""Generates a 256-entry e^x table for lower 8 bits indexing."""
def generate_exp_table(filename = "exp_table.hex"):
    print(f"Generating {filename}")

    with open(filename, "w") as f:
        for idx in range(256):
            x = -10.0 + (idx / 255.0) * 10.0
            val = np.exp(x)
            hex_str = float_to_bf16_hex(val)
            f.write(f"{hex_str}\n")


    print(f"Done! 256 entries written to {filename}")

"""Generates a 128-entry mantissa reciprocal table for 1/M"""
def generate_recip_table(filename = "recip_table.hex"):
    print(f"Generating {filename}")

    with open(filename,"w") as f:
        for m_int in range(128):
            # Step 1: Map index 0..127 to real mantissa float [1.0, 2.0)
            m_float = 1.0 + (m_int / 128.0)

            # Step 2: Compute math reciprocal 1 / M (Range: (0.5, 1.0])
            recip_m = 1.0 / m_float

            frac = (recip_m * 2.0) - 1.0 if recip_m < 1.0 else recip_m -1.0
            

            mant_7bit = int(round(frac * 128.0)) & 0x007F

            f.write(f"{mant_7bit:02x}\n")

    print(f"Done! 128 entries written to {filename}")

if __name__ == "__main__":
    generate_exp_table()
    generate_recip_table()

