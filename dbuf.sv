module dbuf #(
    parameter DATA_WIDTH = 16, // bf16 
    parameter ADDR_WIDTH = 10  // 1024 (degisebilir)
)(
    input logic clk,
    input logic rst_n,

    input  logic [DATA_WIDTH-1:0] rx_data,  
    input  logic                  rx_we,    // AXI write enable sinyali al
    output logic                  rx_full,  // AXI full sinyali ver

    input  logic [ADDR_WIDTH-1:0] read_addr, // FSM hangi adresi okumak istiyor
    output logic [DATA_WIDTH-1:0] read_data, // istenen adresteki veri

    input  logic                  swap_buffers // FSM "hesap bitti, bellekleri takas et" sinyali
);
    logic [DATA_WIDTH-1:0] mem_ping [0:(1<<ADDR_WIDTH)-1]; //ping bellegi
    logic [DATA_WIDTH-1:0] mem_pong [0:(1<<ADDR_WIDTH)-1]; //pong bellegi

    logic write_to_pong; // Durum Kaydedici (0: Ping, 1: Pong)

    logic [ADDR_WIDTH:0] write_addr; //adres sayaci

    always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                write_to_pong <= 1'b0;
                write_addr    <= '0;
            end else begin
                if (swap_buffers) begin
                    write_to_pong <= ~write_to_pong; 
                    write_addr    <= '0;     
                end 
                else if (rx_we && !rx_full) begin
                    if (write_to_pong) begin
                        mem_pong[write_addr] <= rx_data; 
                    end else begin
                        mem_ping[write_addr] <= rx_data; 
                    end
                    write_addr <= write_addr + 1'b1; 
                end
            end
    end

    assign rx_full = write_addr[ADDR_WIDTH];

    always_ff@(posedge clk ) begin
        if (write_to_pong) begin
            read_data <= mem_ping[read_addr];
        end else begin
            read_data <= mem_pong[read_addr];
        end
    end

endmodule