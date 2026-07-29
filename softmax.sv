`timescale 1ns / 1ps

//1) module header & port definitions
module softmax #(
    parameter DATA_WIDTH = 16
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
     
//2) FSM state enumeration & Internal Signals
  //state enumeration
    typedef enum logic [1:0]{
        IDLE                = 2'd0,
        ACCUMULATE          = 2'd1,
        INVERT              = 2'd2,
        DIVIDE_NORMALIZE    = 2'd3
    }state_type;
    
    state_type current_state, next_state;  //state variables  
    
  // pipeline registers
    logic [DATA_WIDTH-1:0]    sum_acc;  //row accumulations 
    logic [DATA_WIDTH-1:0]    inv_sum;   // inverted sum
  
    logic [DATA_WIDTH-1:0] sum_next;
    logic [DATA_WIDTH-1:0] recip_out;
    
  // delayed control flags (to match exp_lut latency)
    logic exp_valid;
    logic exp_last;
    
    logic fifo_empty;
    logic fifo_full;
  
  //internal data lines
    logic [DATA_WIDTH-1:0]    exp_data;
    logic [DATA_WIDTH-1:0]    fifo_dout;
    
    logic [15:0]    element_count;    //FIFO element counter
    
 //3)sub-block instantiations
  
  //LUT exponential call
    exp_lut exp_val(
        .x_in(s_axis_tdata),
        .y_out(exp_data)  
    );
  
  //Row Accumulator Adder
    bf16_add accu(
        .a_in(exp_data),
        .b_in(sum_acc),
        .result_sum_o(sum_next)
    );
  //reciprocal engine (1/summ_acc)
    recip_lut recip_inst(
    .x_in(sum_acc),
    .y_out(recip_out)
    );
  //normalization multiplier
    bf16_mul norm_multiplier(
        .a_in(fifo_dout),            //stored e^x from the buffer
        .b_in(inv_sum),
        .result_mul_o(m_axis_tdata)  // normalized attention weights
     ); 
   //row FIFO buffer memory
     row_fifo mem_buffer(
         .clk(clk),
         .rst_n(rst_n),
         .wr_data(exp_data),
         .wr_en((current_state == ACCUMULATE || current_state == IDLE) && exp_valid),
         .rd_en(current_state == DIVIDE_NORMALIZE && m_axis_tready),
        
         
         .rd_data(fifo_dout),
         .empty(fifo_empty),
         .full(fifo_full)
        
     );
     
   //4)Sequential Register Updates
     always_ff@(posedge clk or negedge rst_n)begin
        if(!rst_n) begin
            current_state <= IDLE;
            sum_acc <= '0;
            inv_sum <= '0;
            exp_last <= '0;
            exp_valid <= '0;
            element_count <= '0;
        end else begin

            current_state <= next_state;      
            
            if (s_axis_tready) begin
                exp_valid <= s_axis_tvalid && s_axis_tready;
                exp_last  <= s_axis_tlast && s_axis_tready;
            end else begin
                exp_valid <= 1'b0;  // Don't register invalid tokens when backpressured
            end
            
            if(current_state == ACCUMULATE ||current_state == IDLE)begin
                if(exp_valid) begin
                    sum_acc          <= sum_next;
                    element_count    <= element_count + 1'b1;
                end             
            end else if(current_state == INVERT) begin
                inv_sum <= recip_out;
                sum_acc <= '0;      // Reset accumulator for the next row       
            end else if(current_state == DIVIDE_NORMALIZE) begin
                if(m_axis_tvalid && m_axis_tready)
                    element_count    <= element_count - 1;
            end else 
                element_count    <= '0;
        end
        

        
     end
     
   //5)Next-State & Control Logic
     always_comb begin
        next_state = current_state;
        
        case(current_state)
        IDLE: begin
            if(s_axis_tvalid && s_axis_tready) next_state = ACCUMULATE;
        end
        
        ACCUMULATE: begin
            if(exp_last) next_state = INVERT;
        end
        
        INVERT: begin
            next_state = DIVIDE_NORMALIZE;
        end
        
        DIVIDE_NORMALIZE: begin
            if(fifo_empty && m_axis_tvalid && m_axis_tready) next_state = IDLE;
        end
        
        default: begin
            next_state = IDLE;
            
        end
        
        endcase
     end
     
     
   //6)Flow COntrol & Output Assignments
      
     assign s_axis_tready = !fifo_full && (current_state == IDLE || current_state == ACCUMULATE);
     
     assign m_axis_tvalid = !fifo_empty && (current_state == DIVIDE_NORMALIZE);
     
     assign m_axis_tlast  = (current_state == DIVIDE_NORMALIZE) &&  (element_count == 16'd1);
    
endmodule
