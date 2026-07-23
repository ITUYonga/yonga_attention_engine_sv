module qk_array#(
    parameter SIZE= 4)
(   input logic clk_i,
    input logic rst_ni,
    input logic [15:0] q_array [0:SIZE-1],
    input logic [15:0] k_array [0:SIZE-1],
    output logic [15:0] sum_out [0:SIZE-1]   
);

    logic [15:0] q_con [0:SIZE] [0:SIZE];
    logic [15:0] k_con [0:SIZE] [0:SIZE];
    logic [15:0] sum_out_con [0:SIZE] [0:SIZE];

    genvar i,j;
    generate 
        for(i=0;i<SIZE;i=i+1) begin :grid_in
            assign q_con[i][0]= q_array[i];
            assign k_con[0][i] = k_array[i];
        end

        for(i=0;i<SIZE;i=i+1) begin :row
            for(j=0;j<SIZE;j=j+1) begin :column

            logic [15:0]current_sum;

            if(i==0)begin 
                assign current_sum = 16'b0;
            end
            else begin 
                assign current_sum = sum_out_con[i][j];
            end

            qk_pe upe(.clk_i(clk_i),
                      .rst_ni(rst_ni),
                      .q_in(q_con[i][j]),
                      .k_in(k_con[i][j]),
                      .pe_sum_in(current_sum),
                      .q_out(q_con[i][j+1]),
                      .k_out(k_con[i+1][j]),
                      .pe_sum_out(sum_out_con[i+1][j])
            );

            end
        end

        for(j=0;j<SIZE;j=j+1) begin :grid_out
            assign sum_out[j] = sum_out_con[SIZE][j];
        end

    endgenerate

endmodule