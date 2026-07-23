module qk_pe(
    input logic clk_i,
    input logic rst_ni,
    input logic [15:0] q_in,
    input logic [15:0] k_in,
    input logic [15:0] pe_sum_in,
    output logic [15:0] q_out,
    output logic [15:0] k_out,
    output logic [15:0] pe_sum_out
);

logic [15:0] mid_sum;
logic [15:0] mid_mul;
logic [15:0] mul_reg;
logic [15:0] sum_reg;

bf16_mul uut(
             .a_in(q_in),
             .b_in(k_in),
             .result_mul_o(mid_mul)   
);

bf16_add dut(
             .a_in(mul_reg),
             .b_in(sum_reg),
             .result_sum_o(mid_sum)
);

always_ff @(posedge clk_i or negedge rst_ni) begin 
    if(!rst_ni) begin 
        q_out <= 16'b0;
        k_out <= 16'b0;
        pe_sum_out <= 16'b0;
        mul_reg <= 16'b0;
        sum_reg <= 16'b0;
    end

    else begin 
        mul_reg <= mid_mul;
        sum_reg <= pe_sum_in;
        pe_sum_out <= mid_sum;
        q_out <= q_in;
        k_out <= k_in;
    end

end


endmodule