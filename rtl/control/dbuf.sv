// FUNC_DBUF : Ping-pong input token buffer
//
//   Purpose:  Holds one full incoming packet (a whole token sequence) in
//             one of two memory banks (mem_ping/mem_pong) while the
//             downstream streamer reads the previous packet out of the
//             other bank, so a new packet can start loading the instant
//             the old one finishes without waiting for the read side to
//             catch up. write_to_pong tracks which bank is currently
//             being written, the read side always looks at the other
//             one. swap_buffers, driven by the write side finishing a
//             packet, flips which bank is which for next time.
//
//   parameters:
//           DATA_WIDTH:   bit width of one bf16 element, should stay 16
//           ADDR_WIDTH:   address width of each bank, bank depth is
//                         2 to the power of ADDR_WIDTH
//
//   inputs:
//           clk:            clock
//           rst_n:          active low reset
//           rx_data:        one element to write in
//           rx_we:          rx_data is valid this cycle, write it
//           read_addr:      address the read side wants to read this
//                           cycle
//           swap_buffers:   this cycle's rx_data is the last element of
//                           the current packet, flip banks for the next
//                           packet after writing it
//   output:
//           rx_full:        write side has run out of room in the
//                           current bank
//           read_data:      element at read_addr from the bank that is
//                           not currently being written, one cycle later
//
//   notes:
//           found and fixed a real bug here, swap_buffers and rx_we are
//           both true on the exact same cycle for a packet's very last
//           element, since swap_buffers is derived from rx_we together
//           with the last flag. The write and the bank/address
//           bookkeeping used to be an if/else against each other, so the
//           swap branch always won and that last element's data was
//           never actually written anywhere, permanently undefined in
//           memory. The two are independent now, both happen on the same
//           cycle

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
                // swap_buffers = rx_we && internal_rx_last, so it is high on
                // the EXACT same cycle as the last element's own write. This
                // used to be an if/else against the data write below, which
                // meant the bank-swap branch always won and the very last
                // element's data was silently dropped -- never written
                // anywhere, leaving that memory word permanently X. The
                // actual data write and the swap/address bookkeeping are
                // independent and must both happen on this cycle.
                if (rx_we && !rx_full) begin
                    if (write_to_pong) begin
                        mem_pong[write_addr] <= rx_data;
                    end else begin
                        mem_ping[write_addr] <= rx_data;
                    end
                end

                if (swap_buffers) begin
                    write_to_pong <= ~write_to_pong;
                    write_addr    <= '0;
                end else if (rx_we && !rx_full) begin
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