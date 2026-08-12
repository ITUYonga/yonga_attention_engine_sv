`timescale 1ns / 1ps

// FUNC_SCALE_MASK : Scales each raw score by 1 over square root of d_k
// and applies causal masking before softmax sees it
//
//   Purpose:  A 3 stage pipeline that multiplies every incoming score by
//             scale_factor through bf16_mul, then either passes the
//             scaled value through unchanged or replaces it with
//             negative infinity if en_scale_mask marks this element as
//             masked, so softmax's own exponential naturally rounds a
//             masked position down to zero without softmax needing to
//             know anything about masking itself. Only one element is
//             ever in flight through the pipeline at a time.
//
//   parameters:
//           DATA_WIDTH:   bit width of one bf16 element, should stay 16
//
//   inputs:
//           clk, rst_n:      clock and active low reset
//           s_axis_tdata:    one raw score
//           s_axis_tvalid:   s_axis_tdata is valid this cycle
//           s_axis_tlast:    this score is the last of its row
//           scale_factor:    1 over square root of d_k, in bf16
//           en_scale_mask:   this element should be masked to negative
//                            infinity instead of scaled through normally
//           m_axis_tready:   softmax can accept a result this cycle
//           ext_stall:       freeze the whole pipeline this cycle,
//                            neither accepting new input nor advancing
//   output:
//           s_axis_tready:   this module can accept one score this cycle
//           m_axis_tdata:    the scaled, or masked, result
//           m_axis_tvalid:   m_axis_tdata is valid this cycle
//           m_axis_tlast:    this result is the last of its row
//
//   notes:
//           found and fixed a real bug here already, the admission
//           condition used to let a second element enter the pipeline
//           just because the final stage register was still empty, even
//           though it was empty only because the first element had not
//           arrived there yet. bf16_mul is a fixed latency pipeline with
//           no stall of its own, so if m_axis_tready then stalled for a
//           couple of cycles, two multiply results could land back to
//           back and the single, non FIFO final register could only keep
//           one, silently dropping a row element. Gating admission on
//           pipe_busy, one element in flight at a time, fixed it at the
//           cost of some throughput

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
    logic [DATA_WIDTH-1:0]      stg1_data;
    logic [DATA_WIDTH-1:0]      stg1_scale;

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
    logic                       stg3_en;
    logic                       pipe_busy;       // one element allowed in flight at a time

    logic                       mul_tag_unused;  //throw away wire

//
// Handshake & backpressure Logic
//
    assign stg3_en = (m_axis_tready || !stg3_vld) && !ext_stall;

    // bf16_mul (and the stg2a/stg2b shift registers that track it) is a
    // fixed-latency, non-stallable pipe: once valid_i pulses, valid_o *will*
    // fire exactly 2 cycles later no matter what, and stg3 is a single
    // register with no FIFO behind it. The original "stg1_en = stg3_en ||
    // !stg1_vld" let a second (and third) element enter stage 1 just
    // because stg3 was still empty, even though stg3 was still empty only
    // because the *first* element hadn't arrived yet. When m_axis_tready
    // later stalls for a couple of cycles (exactly what softmax.sv does
    // while its adder is busy), two mul_valid pulses land back to back and
    // stg3 -- being a single register -- can only keep one, silently
    // dropping an entire row element. Gating admission on pipe_busy so only
    // one element is ever in flight between stg1 and m_axis fixes this at
    // the cost of throughput, which is fine at this row size.
    assign stg1_en = !pipe_busy && !ext_stall;

    assign s_axis_tready = stg1_en;                                  // Upstream ready matches Stage 1 enable state

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_busy <= 1'b0;
        end else if (stg1_en && s_axis_tvalid) begin
            pipe_busy <= 1'b1;
        end else if (m_axis_tvalid && m_axis_tready) begin
            pipe_busy <= 1'b0;
        end
    end

 //
 // Stage 1 Input Capture
 // Adds 1cc latency ==> Timin Closure
 // s_axis_treadt    ==> Clean Handshake

    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n)begin
            stg1_vld         <= '0;
            stg1_tlast       <= '0;
            stg1_en_mask     <= '0;
            stg1_data        <= '0;
            stg1_scale       <= '0;
        end else if(stg1_en) begin
            stg1_vld      <= s_axis_tvalid;
            if(s_axis_tvalid) begin
                stg1_tlast       <= s_axis_tlast;
                stg1_en_mask    <= en_scale_mask;
                stg1_data        <= s_axis_tdata;
                stg1_scale       <= scale_factor;
            end
        end else begin
            // Without this, stg1_vld simply holds its last value while
            // stg1_en is low (pipe_busy blocking further input), staying
            // stuck at 1 for the whole time an element is in flight. Since
            // stg1_vld drives bf16_mul's valid_i directly, bf16_mul (a
            // fixed-latency pipeline with no backpressure of its own) reads
            // that stuck-high level as a fresh operation every single
            // cycle, so valid_o pulses continuously instead of once,
            // triggering repeated phantom captures downstream. stg1_vld
            // must be a genuine one-cycle pulse, so clear it explicitly the
            // cycle after it fires.
            stg1_vld <= 1'b0;
        end
    end






//BF16_MUL Instantiation (2cc latency)
    bf16_mul multiplier_inst(
        .clk_i(clk),
        .rst_ni(rst_n),
    
        .valid_i(stg1_vld),                            //waits 1 cc for stg1_vld
        .a_in({1'b0, stg1_data}),                      // 1 extra tag bit added(as Taha sends clipped data)
        .b_in({1'b0, stg1_scale}),                     // score * scale factor; score = Q*KT     
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
            end else begin
                stg2a_vld     <= stg1_vld;
                stg2a_tlast   <= stg1_tlast;
                stg2a_en_mask <= stg1_en_mask;
            end
        end

    // stg2b: shift every clock (matches mul internal s2 / valid_o)
        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                stg2b_vld <= '0; stg2b_tlast <= '0; stg2b_en_mask <= '0;
            end else begin
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
// stg3 already implements a complete valid/ready held register (stg3_en =
// m_axis_tready || !stg3_vld, so it only advances once the current value
// has been consumed downstream). A second, separately-clocked m_axis_tvalid
// register re-latching stg3_vld on its own "m_axis_tready || !m_axis_tvalid"
// condition duplicates that same hold logic in a second flip-flop that
// isn't guaranteed to track stg3 cycle-for-cycle: whenever m_axis_tready
// stayed high for consecutive cycles, this register could re-present
// stg3_vld=1 as a "new" transfer on more than one of those cycles even
// though stg3 itself hadn't advanced to a new element yet, so softmax
// ended up accepting the same element more than once per row. Driving
    // m_axis directly from stg3 removes the second, redundant register.
    assign m_axis_tvalid = stg3_vld;
    assign m_axis_tdata  = (stg3_en_mask) ? BF16_NEG_INF : stg3_scaled;
    assign m_axis_tlast  = stg3_tlast;

endmodule