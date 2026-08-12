// Bit-exact BF16 golden-model helpers for testbenches only. Not
// synthesizable, not used anywhere in rtl/.
//
// Unlike bf16_test_utils.svh's real_to_bf16/bf16_to_real (round-to-nearest,
// meant for picking/reading back test values that are exact in bf16 to
// begin with), the functions here are direct ports of the actual RTL
// arithmetic in rtl/compute/bf16_mul.sv and rtl/compute/bf16_add.sv: plain
// truncation, no rounding, same bit-for-bit shifts and normalization steps.
// Needed whenever a golden value depends on inputs that are NOT exact in
// bf16 (e.g. RoPE's sin/cos ROM contents), where "compute in real math and
// round once at the end" and "truncate after every single step like the
// hardware does" do not generally agree.

function automatic logic [15:0] bf16_mul_bits(input logic [15:0] a, input logic [15:0] b);
    logic        sign_a, sign_b, final_sign;
    logic [7:0]  expo_a, expo_b;
    logic [7:0]  mant_a, mant_b;
    logic [15:0] final_mant;
    logic signed [9:0] exponent_temp;
    logic        zero_flag;
    begin
        sign_a = a[15];
        sign_b = b[15];
        expo_a = a[14:7];
        expo_b = b[14:7];
        mant_a = (|expo_a) ? {1'b1, a[6:0]} : 8'b0;
        mant_b = (|expo_b) ? {1'b1, b[6:0]} : 8'b0;

        final_sign = sign_a ^ sign_b;
        final_mant = mant_a * mant_b;
        zero_flag  = (expo_a == 8'b0) || (expo_b == 8'b0);
        exponent_temp = $signed({2'b00, expo_a}) + $signed({2'b00, expo_b})
                        - 10'sd127 + (final_mant[15] ? 10'sd1 : 10'sd0);

        if (zero_flag)
            bf16_mul_bits = 16'b0;
        else if (exponent_temp <= 10'sd0)
            bf16_mul_bits = 16'b0;
        else if (exponent_temp >= 10'sd255)
            bf16_mul_bits = {final_sign, 8'hFF, 7'b0};
        else if (final_mant[15])
            bf16_mul_bits = {final_sign, exponent_temp[7:0], final_mant[14:8]};
        else
            bf16_mul_bits = {final_sign, exponent_temp[7:0], final_mant[13:7]};
    end
endfunction

function automatic logic [15:0] bf16_add_bits(input logic [15:0] a, input logic [15:0] b);
    logic        sign_a, sign_b;
    logic [7:0]  expo_a, expo_b;
    logic [7:0]  mant_a, mant_b;
    logic [7:0]  expo_dif, expo_shared;
    logic [7:0]  lined_mant_a, lined_mant_b;
    logic signed [9:0] signed_a, signed_b, sum;
    logic [8:0]  abs_sum;
    logic        final_sign;
    logic [8:0]  final_sum;
    logic [3:0]  norm_shift;
    logic [6:0]  normalized_mant;
    logic signed [9:0] normalized_expo;
    begin
        sign_a = a[15];
        sign_b = b[15];
        expo_a = a[14:7];
        expo_b = b[14:7];
        mant_a = (|expo_a) ? {1'b1, a[6:0]} : 8'b0;
        mant_b = (|expo_b) ? {1'b1, b[6:0]} : 8'b0;

        if (expo_a > expo_b) begin
            expo_dif     = expo_a - expo_b;
            expo_shared  = expo_a;
            lined_mant_a = mant_a;
            lined_mant_b = (expo_dif >= 8'd8) ? 8'b0 : (mant_b >> expo_dif);
        end else begin
            expo_dif     = expo_b - expo_a;
            expo_shared  = expo_b;
            lined_mant_b = mant_b;
            lined_mant_a = (expo_dif >= 8'd8) ? 8'b0 : (mant_a >> expo_dif);
        end

        signed_a = sign_a ? -$signed({2'b00, lined_mant_a}) : $signed({2'b00, lined_mant_a});
        signed_b = sign_b ? -$signed({2'b00, lined_mant_b}) : $signed({2'b00, lined_mant_b});

        sum = signed_a + signed_b;
        final_sign = sum[9];
        abs_sum = final_sign ? $unsigned(-sum) : $unsigned(sum);
        final_sum = abs_sum[8:0];

        if (final_sum == 9'b0) begin
            bf16_add_bits = 16'b0;
        end else if (final_sum[8]) begin
            if (expo_shared >= 8'd254)
                bf16_add_bits = {final_sign, 8'hFF, 7'b0};
            else
                bf16_add_bits = {final_sign, expo_shared + 8'd1, final_sum[7:1]};
        end else begin
            if (final_sum[7]) begin
                norm_shift = 4'd0; normalized_mant = final_sum[6:0];
            end else if (final_sum[6]) begin
                norm_shift = 4'd1; normalized_mant = {final_sum[5:0], 1'b0};
            end else if (final_sum[5]) begin
                norm_shift = 4'd2; normalized_mant = {final_sum[4:0], 2'b0};
            end else if (final_sum[4]) begin
                norm_shift = 4'd3; normalized_mant = {final_sum[3:0], 3'b0};
            end else if (final_sum[3]) begin
                norm_shift = 4'd4; normalized_mant = {final_sum[2:0], 4'b0};
            end else if (final_sum[2]) begin
                norm_shift = 4'd5; normalized_mant = {final_sum[1:0], 5'b0};
            end else if (final_sum[1]) begin
                norm_shift = 4'd6; normalized_mant = {final_sum[0], 6'b0};
            end else begin
                norm_shift = 4'd7; normalized_mant = 7'b0;
            end

            normalized_expo = $signed({2'b00, expo_shared}) - $signed({6'b0, norm_shift});

            if (normalized_expo <= 10'sd0)
                bf16_add_bits = 16'b0;
            else
                bf16_add_bits = {final_sign, normalized_expo[7:0], normalized_mant};
        end
    end
endfunction

// exp_lut.sv and recip_lut.sv's actual ROM/LUT lookup (lut_rom[x_in[7:0]],
// mant_rom[mant_in]) is deliberately NOT wrapped in a function here: passing
// an unpacked array into a subroutine is legal SystemVerilog but not
// portable across every simulator's automatic-function support, and the
// array index is one line either way. exp_lut_special() and
// recip_lut_exponent() below cover everything EXCEPT that single indexed
// read, which the caller (which already has the ROM array in scope) does
// directly. See tb_top.sv's build_golden_model() for the full assembly.

// Direct port of rtl/softmax/exp_lut.sv's special-case logic (the branches
// that do not touch the ROM). is_special=1 means special_val is the final
// answer; is_special=0 means the caller must still look up
// exp_table_rom[x_in[7:0]] itself.
task automatic exp_lut_special(
    input  logic [15:0] x_in,
    output logic         is_special,
    output logic [15:0]  special_val
);
    logic       sign;
    logic [7:0] expo;
    begin
        sign = x_in[15];
        expo = x_in[14:7];
        is_special  = 1'b1;
        special_val = 16'h0000;
        if (x_in == 16'h0000 || x_in == 16'h8000)
            special_val = 16'h3F80;
        else if (sign && (expo >= 8'd133))
            special_val = 16'h0000;
        else if (!sign && (expo >= 8'd133))
            special_val = 16'h7F80;
        else
            is_special = 1'b0;
    end
endtask

// Direct port of rtl/softmax/recip_lut.sv's sign/exponent logic (everything
// except the mant_rom[mant_in] lookup, which the caller does itself). Also
// returns mant_in so the caller knows which ROM address to read.
task automatic recip_lut_exponent(
    input  logic [15:0] x_in,
    output logic         is_special,
    output logic [15:0]  special_val,
    output logic [6:0]   mant_in,
    output logic         sign_in,
    output logic [7:0]   expo_out
);
    logic [7:0] expo_in;
    begin
        sign_in = x_in[15];
        expo_in = x_in[14:7];
        mant_in = x_in[6:0];
        expo_out = (mant_in == 7'b0) ? (8'd254 - expo_in) : (8'd253 - expo_in);
        if (expo_in == 8'h00) begin
            is_special  = 1'b1;
            special_val = {sign_in, 8'hFF, 7'b0};
        end else begin
            is_special  = 1'b0;
            special_val = 16'h0000;
        end
    end
endtask
