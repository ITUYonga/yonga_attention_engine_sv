`timescale 1ns / 1ps


//1) module header & port definitions
module scale_mask #(
    parameter DATA_WIDTH = 16
  //parameter DATA_WIDTH_f32 = 32
)(
  //port definitions
  //basic ports
    input  logic clk,
    input  logic rst_n,
  
  //AXI-4 Slave(PE Array  --> scale_mask, from Taha)
    input  logic [DATA_WIDTH-1:0]    s_axis_tdata,   // 16 bit bf16 score
    input  logic                     s_axis_tvalid,  // data valid flag
    output logic                     s_axis_tready,  // ready flag
    input  logic                     s_axis_tlast,   // end of row indicator   
  
  //Configuration & Control(from Taha)
    input  logic [DATA_WIDTH-1:0]    scale_factor,   //1/sqrt(d_k)
    input  logic                     en_scale_mask,  
  
  //AXI-4 Master (scale_mask --> softmax)
    output logic [DATA_WIDTH-1:0]    m_axis_tdata,   // scaled & masked bf_16 score
    output logic                     m_axis_tvalid,  // output valid flag
    input  logic                     m_axis_tready,  // downstream ready flag
    output logic                     m_axis_tlast    // end of row indicator     
    );

//2) constants and internal wires
    localparam logic [DATA_WIDTH-1:0]    BF16_NEG_INF = 'hFF80; // e^-infinity = 0
    
    logic [DATA_WIDTH-1:0]      scaled_score; //output of the multiplier
    logic [DATA_WIDTH-1:0]      masked_score; //output of the masking multiplexer
    
//3) datapath instantiation & combinational logic
    bf16_mul multiplier_inst(
        .a_in(s_axis_tdata),
        .b_in(scale_factor),
        .result_mul_o(scaled_score)
    );
    always_comb begin    
        if(en_scale_mask) 
            masked_score = BF16_NEG_INF;  //overwrite -infinity  
        else    
            masked_score = scaled_score;  //pass the valid score            
    end
                           
           
//4) Stream Flow-Control Logic
    assign s_axis_tready = m_axis_tready;
    
//5)Sequential Output Pipeline Registers

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin        //all master AXI variables set to 0
            m_axis_tdata <= 0;
            m_axis_tvalid <= 0;
            m_axis_tlast <= 0;
        end else if(s_axis_tready) begin
            m_axis_tvalid <= s_axis_tvalid; 
            m_axis_tlast  <= s_axis_tlast && s_axis_tvalid; // Only high when valid
            if(s_axis_tvalid) begin     // if valid flag has received from slave, data and last are transferred from slave to master
                m_axis_tdata <= masked_score;
            end
        end    
    end
     

endmodule


