// Shared BF16 <-> real helpers for testbenches only. Not synthesizable,
// not used anywhere in rtl/. Handles normal-range values; no inf/NaN/subnormal
// support since none of the test stimuli need it.

function automatic real bf16_to_real(input logic [15:0] b);
    logic        sign;
    logic [7:0]  exp;
    logic [6:0]  mant;
    real         sign_r, mant_r;
    begin
        sign = b[15];
        exp  = b[14:7];
        mant = b[6:0];
        sign_r = sign ? -1.0 : 1.0;
        if (exp == 8'h00 && mant == 7'h00) begin
            bf16_to_real = 0.0;
        end else begin
            mant_r = 1.0 + (real'(mant) / 128.0);
            bf16_to_real = sign_r * mant_r * (2.0 ** (real'(exp) - 127.0));
        end
    end
endfunction

function automatic logic [15:0] real_to_bf16(input real r);
    int    sign;
    real   ar, mant;
    int    exp_val;
    int    mant_bits;
    logic [7:0] exp_field;
    begin
        if (r == 0.0) begin
            real_to_bf16 = 16'h0000;
        end else begin
            sign = (r < 0.0) ? 1 : 0;
            ar   = (r < 0.0) ? -r : r;

            exp_val = $floor($ln(ar) / $ln(2.0));
            mant    = ar / (2.0 ** exp_val) - 1.0;

            if (mant < 0.0) begin
                exp_val = exp_val - 1;
                mant    = ar / (2.0 ** exp_val) - 1.0;
            end else if (mant >= 1.0) begin
                exp_val = exp_val + 1;
                mant    = ar / (2.0 ** exp_val) - 1.0;
            end

            mant_bits = $floor(mant * 128.0 + 0.5);
            if (mant_bits == 128) begin
                mant_bits = 0;
                exp_val   = exp_val + 1;
            end

            exp_field = exp_val + 127;
            real_to_bf16 = {sign[0], exp_field, mant_bits[6:0]};
        end
    end
endfunction
