`timescale 1ns / 1ps

//modified with claude
//added 3rd pipeline stage
//

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
    output logic                     m_axis_tlast,    // end of row indicator     

  //***
  //Taha Channel Control Flag(!!NEW)
    input  logic                     ext_stall      // 1 = Freeze channel, 0 = Normal
    );

//2) constants and internal wires
    //localparam logic [DATA_WIDTH:0]    BF16_NEG_INF = {1'b0, 16'hFF80}; // e^-infinity = 0
    localparam logic [DATA_WIDTH-1:0]  BF16_NEG_INF = 16'hFF80;
    
    logic [DATA_WIDTH-1:0]      mul_result;     //combinational output of the multiplier
    logic                       mul_valid;      //valid from mul module instantiation

    //output of the masking multiplexer
    
    
//Stage 1 Pipeline Registers
    logic                       stg1_vld;
    logic                       stg1_tlast;
    logic                       stg1_en_mask;

//Stage 2(2 internal stages) Pipeline Registers
    logic                       stg2a_vld, stg2a_tlast, stg2a_en_mask;
    logic                       stg2b_vld, stg2b_tlast, stg2b_en_mask;

//Stage 3 Pipeline Registers
    logic                       stg3_vld;
    logic [DATA_WIDTH-1:0]      stg3_scaled;
    logic                       stg3_tlast;
    logic                       stg3_en_mask;

// Pipeline Enable / Handshake Logic Wires
    logic                       stg1_en;
    logic                       stg2_en;
    logic                       stg3_en;

    logic                       mul_tag_unused;  //throw away wire
    
//
// Handshake & backpressure Logic
//
    assign stg3_en = (m_axis_tready || !stg3_vld) && !ext_stall;
    assign stg2_en = (stg3_en || !stg2b_vld)     && !ext_stall;  // Stage 2 enables IF downstream is ready OR empty, AND friend is not stalling
    assign stg1_en = (stg2_en || !stg1_vld)       && !ext_stall;            // Stage 1 enables IF Stage 2 enables OR empty, AND friend is not stalling
     
    assign s_axis_tready = stg1_en;                                  // Upstream ready matches Stage 1 enable state

 //
 // Stage 1 Input Capture
 // Adds 1cc latency ==> Timin Closure
 // s_axis_treadt    ==> Clean Handshake

    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n)begin
            stg1_vld         <= '0;
            stg1_tlast       <= '0;
            stg1_en_mask     <= '0;
        end else if(stg1_en) begin
            stg1_vld      <= s_axis_tvalid;
            if(s_axis_tvalid) begin
                stg1_tlast       <= s_axis_tlast;
                stg1_en_mask    <= en_scale_mask; 
            end
        end
    end






//BF16_MUL Instantiation (2cc latency)
    bf16_mul multiplier_inst(
        .clk_i(clk),
        .rst_ni(rst_n),
    
        .valid_i(stg1_vld),                            //waits 1 cc for stg1_vld
        .a_in({1'b0, s_axis_tdata}),                   // 1 extra tag bit added(as Taha sends clipped data)
        .b_in({1'b0, scale_factor}),                   // score * scale factor; score = Q*KT     
        .result_mul_o({mul_tag_unused, mul_result}),    
        //discard the tag bit
    
        .valid_o(mul_valid)
    );

    //
    // Stage 2 Registers (Capture bf16_mul Pipeline State)
    //

    // stg2a: shift every clock (matches mul internal s1)
        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                stg2a_vld <= '0; stg2a_tlast <= '0; stg2a_en_mask <= '0;
            end else if (stg2_en) begin
                stg2a_vld     <= stg1_vld;
                stg2a_tlast   <= stg1_tlast;
                stg2a_en_mask <= stg1_en_mask;
            end
        end

    // stg2b: shift every clock (matches mul internal s2 / valid_o)
        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                stg2b_vld <= '0; stg2b_tlast <= '0; stg2b_en_mask <= '0;
            end else if (stg2_en) begin
                stg2b_vld     <= stg2a_vld;
                stg2b_tlast   <= stg2a_tlast;
                stg2b_en_mask <= stg2a_en_mask;
            end
        end
                           
           

//
//Stage 3 Registers (Final: Apply Masking to the bf16_mul Output Stage)
//

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stg3_vld <= '0; stg3_tlast <= '0;
            stg3_en_mask <= '0; stg3_scaled <= '0;
        end else if (stg3_en) begin
            stg3_vld <= mul_valid;
            if (mul_valid) begin
                stg3_tlast   <= stg2b_tlast;    // was stg2_tlast
                stg3_en_mask <= stg2b_en_mask;  // was stg2_en_mask
                stg3_scaled  <= mul_result;
            end
        end
    end
     
//
// Output Stage (AXI-4 Master)
//

    always_ff @(posedge clk or negedge rst_n)begin
        if(!rst_n) begin
            m_axis_tdata  <=  '0;
            m_axis_tvalid <= '0;
            m_axis_tlast  <= '0;
        end else if (m_axis_tready || !m_axis_tvalid) begin 
            m_axis_tvalid <= stg3_vld;
            if(stg3_vld) begin
                m_axis_tdata  <= (stg3_en_mask) ? BF16_NEG_INF : stg3_scaled;
                m_axis_tlast  <= stg3_tlast;            
            end
        end
    end

endmodule


