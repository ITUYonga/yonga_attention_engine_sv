module mod_a_output_router #(parameter DATA_WIDTH = 16) (
    input  logic                  mod_a_valid,
    input  logic [DATA_WIDTH-1:0] mod_a_data,
    input  logic                  mod_a_tag, 
    output logic                  mod_a_ready,
    // Modül C'ye giden (Softmax)
    output logic                  c_valid,
    output logic [DATA_WIDTH-1:0] c_data,
    input  logic                  c_ready,
    // Modül B Hakemine dönen (W0)
    output logic                  b_valid,
    output logic [DATA_WIDTH-1:0] b_data,
    input  logic                  b_ready
);
    assign c_data  = mod_a_data;
    assign b_data  = mod_a_data;
    assign c_valid = mod_a_valid && (mod_a_tag == 1'b0);
    assign b_valid = mod_a_valid && (mod_a_tag == 1'b1);
    assign mod_a_ready = (mod_a_tag == 1'b0) ? c_ready : b_ready;
endmodule