`timescale 1ns / 1ps

module recip_lut (
    input  logic        clk,
    input  logic [15:0] x_in,   // 16-bit bf16 sum_acc
    output logic [15:0] y_out   // 16-bit bf16 reciprocal 1/sum_acc (1-cycle latency)
);

    localparam string HEX_FILE_PATH = "C:/YONGA/fpt/softmax/recip_table.hex";

    logic       sign_in;
    logic [7:0] exp_in;
    logic [6:0] mant_in;

    assign sign_in = x_in[15];
    assign exp_in  = x_in[14:7];
    assign mant_in = x_in[6:0];

    // ------------------------------------------------------------------------
    // 1) Exponent Transformation: E_out = 254 - E_in
    // ------------------------------------------------------------------------
    logic [7:0] exp_out;
    assign exp_out = 8'd254 - exp_in;

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

    // Registered output for 1 clock cycle latency
    always_ff @(posedge clk) begin
        y_out <= recip_comb;
    end

endmodule