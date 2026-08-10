`timescale 1ns / 1ps

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
    
    always_ff@(posedge clk)begin
        y_out <= exp_comb;
    end

endmodule