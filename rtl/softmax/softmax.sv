`timescale 1ns / 1ps

//
//1) module header & port definitions
//
module softmax #(
    parameter int DATA_WIDTH = 16,
    parameter int MAX_ROW_LEN = 1024
    )(
  //port definitions
  //system signals
    input  logic clk,
    input  logic rst_n,

  //AXI-4 Slave(scale_mask --> softmax)
    input  logic [DATA_WIDTH-1:0]    s_axis_tdata,   // 16 bit bf16 score
    input  logic                     s_axis_tvalid,  // data valid flag
    output logic                     s_axis_tready,  // ready flag
    input  logic                     s_axis_tlast,   // end of row indicator   


  //AXI-4 Master (softmax --> Score x V)
    output logic [DATA_WIDTH-1:0]    m_axis_tdata,   // scaled & masked bf_16 score
    output logic                     m_axis_tvalid,  // output valid flag
    input  logic                     m_axis_tready,  // downstream ready flag
    output logic                     m_axis_tlast    // end of row indicator     
     );

//     
//2) Internal Register Signals & Control Signal Declerations 
//


// High-Level Phase Controller States
  typedef enum logic [1:0] {
    IDLE             = 2'd0,
    ACCUMULATE       = 2'd1,
    INVERT           = 2'd2,
    DIVIDE_NORMALIZE = 2'd3
  } state_type;

  state_type current_state, next_state;

  localparam int COUNT_WIDTH = $clog2(MAX_ROW_LEN) + 1; //Dynamically Sized Counter

  //Stage 1: Input Capture & exp_lut stage
  logic [DATA_WIDTH-1:0]    stage1_data;
  logic                     stage1_valid;
  logic                     stage1_last;
  logic [DATA_WIDTH-1:0]    stage1_exp_data;

  //Stage 2: Reduction/Accumulation & Buffer Writing Stage
  logic [DATA_WIDTH-1:0]    sum_acc;          //used in putting the exp info into the fifp
  logic                     add_busy;         //high while waiting adder for 2cc

  logic [DATA_WIDTH:0]    add_result;
  logic                     add_valid_o;

  //Stage 3: Inversion/ Reciprocal Stage
  logic [DATA_WIDTH-1:0]    recip_out;       // Output from recip_lut
  logic [DATA_WIDTH-1:0]    inv_sum_reg;     //only holds the reciprocal value


  //Stage 4: Normalization/Multiply & Output Stage
  logic [DATA_WIDTH-1:0]    stage4_data;
  logic                     stage4_valid;
  logic                     stage4_last;

  logic [DATA_WIDTH:0]      mul_result;
  logic                     mul_valid_o;

  //Flow Contol and FIFO Internal Signals
  logic                      fifo_wr_en, fifo_rd_en;
  logic                      fifo_empty, fifo_full; 
  logic [DATA_WIDTH-1:0]     fifo_dout;
  logic [COUNT_WIDTH-1:0]    element_count;

//parallel delay registers for the last flag
  logic last_d1, last_d2;
//
//3) Sub-block Instantaitons & Stage 1 input logic
//

  //A)exponential LUT (stage1)

  exp_lut exp_val(
    .clk(clk),
    .x_in(stage1_data),
    .y_out(stage1_exp_data)
  );

  //B)row accumulator adder(stage2)
  bf16_add accu(
    .clk_i(clk),
    .rst_ni(rst_n),
    .valid_i(stage1_valid),

    .a_in({1'b0, stage1_exp_data}),
    .b_in({1'b0, sum_acc}),
    .result_sum_o(add_result),
    .valid_o(add_valid_o)
  );

//C) recip lut(stage3)
  recip_lut recip_inst(
    .clk(clk),
    .x_in(sum_acc),
    .y_out(recip_out)
  );

//D) normalization multiplier(stage4)
 bf16_mul norm_multiplier(
    .clk_i(clk),
    .rst_ni(rst_n),
    .valid_i(fifo_rd_en),
    .a_in({1'b0, fifo_dout}),          //stored e^x from the buffer
    .b_in({1'b0, inv_sum_reg}),
    .result_mul_o(mul_result),                 // normalized attention weights
    .valid_o(mul_valid_o)

  ); 


//E) row fifp buffer
  assign fifo_wr_en = stage1_valid && (current_state == ACCUMULATE || current_state == IDLE);
  assign fifo_rd_en = (current_state == DIVIDE_NORMALIZE) && m_axis_tready && !fifo_empty;

  row_fifo mem_buffer (
    .clk    (clk),
    .rst_n  (rst_n),
    .wr_data(stage1_exp_data),
    .wr_en  (fifo_wr_en),
    .rd_en  (fifo_rd_en),
    .rd_data(fifo_dout),
    .empty  (fifo_empty),
    .full   (fifo_full)
  );

//Stages:

//Stage 1
  always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage1_data  <= '0;
            stage1_valid <= 1'b0;
            stage1_last  <= 1'b0;
        end else begin
          // Pulse stage1_valid strictly on handshaking
          if (s_axis_tvalid && s_axis_tready) begin
              stage1_valid <= 1'b1;
              stage1_data  <= s_axis_tdata;
              stage1_last  <= s_axis_tlast;
          end else begin
              stage1_valid <= 1'b0;
          end
        end
  end

//
//4) Sequential Register Updates & 2cc gate control
//


  always_ff @(posedge clk or negedge rst_n)begin
    if (!rst_n) begin
      current_state <= IDLE;
      sum_acc       <= '0;
      inv_sum_reg   <= '0;
      element_count <= '0;
      add_busy      <= 1'b0;
      last_d1       <= 1'b0;
      last_d2       <= 1'b0;
    end else begin
    //State register updates
      current_state <= next_state;
      //2cc accumulator gate logic
      //
      // add_busy used to be set from stage1_valid, which is itself already
      // a registered, one-cycle-delayed copy of "an element was accepted."
      // That left one full cycle, right after acceptance, where add_busy
      // was still 0 and s_axis_tready (gated only on !add_busy) stayed
      // high, letting a second element sneak in before add_busy ever
      // caught up -- a 4-element row would accumulate 5 additions. Setting
      // add_busy directly from this cycle's own accept condition closes
      // that gap immediately instead of one cycle late.
      if (s_axis_tvalid && s_axis_tready) begin
        add_busy <= 1'b1;                     //adder working, freeze upstream
      end else if (add_valid_o) begin
        add_busy <= 1'b0;                     // unfreeze when adder result is ready
      end

      if (stage1_valid) begin
        last_d1 <= stage1_last;  // Cycle T+1
      end

      // Shift second cycle tag along with addition completion
      if (add_busy) begin
          last_d2 <= last_d1;       // Cycle T+2 (Aligned with add_valid_o)
      end

// Accumulator update (Pass 1)
      if (add_valid_o) begin
        sum_acc <= add_result[DATA_WIDTH-1:0];  
      end

      // Element Counter Management (Single unified driver)
      //
      // Decrementing on mul_valid_o (instead of on the actual m_axis
      // handshake, m_axis_tvalid && m_axis_tready) counts one cycle too
      // early: mul_valid_o is the *input* to the m_axis output register,
      // which only becomes m_axis_tvalid on the following cycle. That skew
      // made m_axis_tlast (= element_count == 1) assert on the
      // second-to-last output instead of the true last one, which in turn
      // made the DIVIDE_NORMALIZE exit condition below fire one output
      // early, discarding the final row element. Keying the decrement to
      // the real output handshake keeps element_count aligned with which
      // output element is actually on the bus this cycle.
      if (add_valid_o) begin
        element_count <= element_count + 1'b1;
      end else if (current_state == DIVIDE_NORMALIZE && m_axis_tvalid && m_axis_tready) begin
        element_count <= element_count - 1'b1;
      end else if (current_state == IDLE) begin
        element_count <= '0;
      end

      //inversion latch
      if(current_state == INVERT) begin
        inv_sum_reg <= recip_out;               //store 1/sum for Pass 2
        sum_acc     <= '0;       // Reset accumulator for next row
        last_d1     <= 1'b0;
        last_d2     <= 1'b0;
      end 
    end
  end

//
//5)Comb. next State Logic
//

always_comb begin
  next_state = current_state;

  case (current_state)
    IDLE: begin
        if (s_axis_tvalid && s_axis_tready) begin
            next_state = ACCUMULATE;
        end
    end

    ACCUMULATE: begin
        // Move to INVERT only after the final element finishes 2cc adder stage
        if (add_valid_o && last_d2) begin
            next_state = INVERT;
        end
    end

    INVERT: begin
        // 1 cycle to latch 1/sum into inv_sum_reg
        next_state = DIVIDE_NORMALIZE;
    end

    DIVIDE_NORMALIZE: begin
        // fifo_empty only means every element has been *read out of the
        // FIFO*, not that every element has finished draining through
        // bf16_mul's multi-cycle pipeline and reached m_axis -- the last
        // couple of reads are still in flight when the FIFO goes empty, so
        // gating on fifo_empty here used to return to IDLE while real
        // output elements were still in the pipe, dropping them. m_axis_tlast
        // (now correctly aligned, see element_count above) already marks the
        // true final output handshake, so use that instead.
        if (m_axis_tlast && m_axis_tvalid && m_axis_tready) begin
            next_state = IDLE;
        end
    end

    default: begin
        next_state = IDLE;
    end
  endcase
    
end

//
//6)flow control and output assignments
//
// Inbound ready: High in Pass 1 if FIFO has space and accumulator isn't busy
  assign s_axis_tready = !fifo_full && !add_busy &&
                         (current_state == IDLE || current_state == ACCUMULATE);

// Outbound stream assignments (fed directly by bf16_mul pipeline output)
  always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
          m_axis_tvalid <= '0;
          m_axis_tdata  <= '0;
      end else if (m_axis_tready || !m_axis_tvalid) begin
          m_axis_tvalid <= mul_valid_o && (current_state == DIVIDE_NORMALIZE);
          m_axis_tdata  <= mul_result[DATA_WIDTH-1:0];
      end
  end

// Assert last when the final element is streaming out of Pass 2
  assign m_axis_tlast  = (current_state == DIVIDE_NORMALIZE) && (element_count == 'd1);



endmodule