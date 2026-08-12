`timescale 1ns / 1ps

module recip_lut #(
    parameter string HEX_FILE_PATH = "rtl/softmax/recip_table.hex"
)(
    input  logic        clk,
    input  logic [15:0] x_in,   // 16-bit bf16 sum_acc
    output logic [15:0] y_out   // 16-bit bf16 reciprocal 1/sum_acc (1-cycle latency)
);


    logic       sign_in;
    logic [7:0] exp_in;
    logic [6:0] mant_in;

    assign sign_in = x_in[15];
    assign exp_in  = x_in[14:7];
    assign mant_in = x_in[6:0];

    // ------------------------------------------------------------------------
    // 1) Exponent Transformation: E_out = 254 - E_in
    //
    // This is only correct when the input mantissa is exactly 0 (input is
    // an exact power of two, so 1/(1.mantissa) = 1.0 exactly, needing no
    // renormalization). For every other input, 1/(1.mantissa) lands in
    // (0.5, 1), which after renormalizing back into the [1, 2) range needs
    // one extra exponent decrement. Skipping that made every reciprocal of
    // a non-power-of-two input come out exactly 2x too large (e.g. 1/3.0
    // computed as ~0.667 instead of ~0.333) even though the ROM's mantissa
    // values already assume this extra decrement was applied.
    // ------------------------------------------------------------------------
    logic [7:0] exp_out;
    assign exp_out = (mant_in == 7'b0) ? (8'd254 - exp_in) : (8'd253 - exp_in);

    // ------------------------------------------------------------------------
    // 2) Mantissa ROM Lookup
    // ------------------------------------------------------------------------
    logic [6:0] mant_rom [0:127];

    initial begin
        $readmemh(HEX_FILE_PATH, mant_rom);
    end

    logic [6:0] mant_out;
    assign mant_out = mant_rom[mant_in];

    // ------------------------------------------------------------------------
    // 3) Reconstruct BF16 Output
    // ------------------------------------------------------------------------
    logic [15:0] recip_comb;

    always_comb begin
        if (exp_in == 8'h00) begin
            // Divide-by-Zero or Subnormal -> Return Infinity
            recip_comb = {sign_in, 8'hFF, 7'b0}; 
        end else begin
            // Concatenate: {sign, new_exponent, inverted_mantissa}
            recip_comb = {sign_in, exp_out, mant_out};
        end
    end

    // Purely combinational: softmax.sv latches inv_sum_reg <= recip_out on
    // the very cycle sum_acc first holds its final row total (the INVERT
    // state), assuming both are available together. A registered y_out
    // here used to make recip_out lag x_in by 1 cycle, so inv_sum_reg
    // captured the reciprocal of the PREVIOUS (incomplete) sum_acc instead
    // of the final one -- the same latency-vs-consumer mismatch already
    // found and fixed in exp_lut.sv.
    assign y_out = recip_comb;

endmodule