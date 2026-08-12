`timescale 1ns / 1ps

// FUNC_EXP_LUT : Looks up e to the power of x for one bf16 value
//
//   Purpose:  softmax needs an exponential of every scaled score, this
//             does it with a small ROM instead of an iterative
//             approximation, trading a little accuracy for a pipeline
//             that never stalls waiting on convergence. Zero is handled
//             specially (result exactly 1.0), and values far enough
//             negative or positive saturate to 0 or infinity instead of
//             indexing into the table, since the table only really
//             covers the useful middle range.
//
//   parameters:
//           HEX_FILE_PATH:   path to the hex file $readmemh loads the
//                            256 entry lookup table from
//
//   inputs:
//           clk:      clock
//           x_in:     one bf16 value, the exponent's argument
//   output:
//           y_out:    e to the power of x_in in bf16, purely combinational
//
//   notes:
//           the lookup itself only ever uses the low 8 bits of x_in
//           (exponent's low bit plus the full mantissa), the sign and
//           high exponent bits are only used for the special case checks
//
//           this must stay purely combinational, softmax.sv feeds
//           stage1_valid and this module's output into bf16_add's
//           valid_i/a_in on the same cycle as if both described the same
//           element. A registered y_out here was tried once and made the
//           exponential lag its own input by a cycle, so every
//           accumulate silently summed the previous element's value
//           instead of its own

module exp_lut #(
    parameter string HEX_FILE_PATH = "rtl/softmax/exp_table.hex"
)(
    input  logic              clk,
    input  logic    [15:0]    x_in,
    output logic    [15:0]    y_out
);

    logic          sign;
    logic [7:0]    exp;

    assign sign    = x_in[15];
    assign exp     = x_in[14:7];

    logic [15:0] lut_rom [0:255];
    
    initial begin
        $readmemh(HEX_FILE_PATH, lut_rom);
    end
    
    logic [15:0]    lookup_val;
    logic [15:0]    exp_comb;  //output register
    
    
    assign lookup_val = lut_rom[x_in[7:0]];  //lower 8 bits (LSB of the Exp. + Mantisssa) are used for addressing
    
    always_comb begin
        if(x_in == 16'h0000 || x_in == 16'h8000)begin  // x == +0 or -0
            exp_comb = 16'h3F80;                       //exp = 1
                        
        end else if(sign && (exp >= 8'd133))begin   // x < -88
            exp_comb = 16'h0000;                    //underflow => exp = 0.0
            
        end else if(!sign && (exp >= 8'd133))begin  // x > 8
            exp_comb = 16'h7F80;                    //overflow => exp = infinity
        end else begin
            exp_comb = lookup_val;
        end
    end
    
    // Purely combinational: softmax.sv feeds stage1_valid (same-cycle as
    // stage1_data) straight into bf16_add's valid_i alongside stage1_exp_data
    // as if both came from the same instant. A registered y_out here used to
    // make stage1_exp_data lag stage1_data by 1 cycle, so every accumulate
    // silently summed the *previous* element's exp value instead of its own,
    // and the row_fifo entry written alongside it was equally off by one.
    // This was invisible whenever every element in a row shared the same
    // value (softmax(all-zero row) -> uniform 0.25, exactly what the
    // "no mask" test used), but broke as soon as one element's value
    // actually differed from its neighbors (the masked-element test).
    assign y_out = exp_comb;

endmodule