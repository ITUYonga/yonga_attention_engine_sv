`timescale 1ns / 1ps

// FUNC_URAM_PINGPONG_CONTROLLER : Drives the write and read addressing
// for the Q, K, and V URAM banks, one bank pair per matrix
//
//   Purpose:  Writing side just follows Modül B's own streaming output,
//             counting up through MATRIX_SIZE elements and flipping to
//             the other ping-pong bank once a whole matrix has landed, so
//             a new matrix can start writing while the array is still
//             reading the previous one out of the other bank. Reading
//             side is the more involved half, it walks the stored,
//             token-major Q/K data back out in the depth-major order
//             qk_array actually wants (see its own header), one full
//             sweep of every row at each depth index before moving to the
//             next depth index, then once softmax produces its first
//             result it switches to reading V out straight through in
//             plain order for the score times V pass.
//
//   parameters:
//           ADDR_WIDTH:    address width of each URAM bank
//           MATRIX_SIZE:   total elements in one full Q/K/V matrix
//           D_MODEL:       elements per token/row, used to reshape the
//                          flat MATRIX_SIZE buffer into depth-major order
//
//   inputs:
//           clk, rst_n:     clock and active low reset
//           write_valid:    one element (shared across Q, K, V) is valid
//                           this cycle
//           write_last:     unused, see notes
//           softmax_valid:  softmax has produced its first result,
//                           switch from the Q/K sweep to reading V
//           qk_data_valid:  this cycle's Q and K URAM reads are both
//                           valid
//           qk_ready:       the array actually accepted this cycle's Q/K
//                           pair
//   output:
//           waddr:          write address shared by the Q, K, and V
//                           banks, top bit selects the ping-pong bank
//           q_ren, k_ren, v_ren:
//                           read enable for each bank
//           q_raddr, k_raddr, v_raddr:
//                           read address for each bank
//
//   notes:
//           write_last is not used for deciding when a matrix is
//           complete, only wr_ptr reaching MATRIX_SIZE-1 is, since
//           write_last used to pulse once per token instead of once per
//           whole matrix and closed out a package far too early
//
//           found and fixed a real bug in the read side's last address:
//           the state machine used to leave ST_READ_QK the instant it
//           issued its final address, before anything confirmed that
//           address's result had actually been accepted downstream.
//           Every earlier address relied on the next address's own hold
//           check to confirm it landed, the last address had no next one
//           to confirm it. final_addr_issued now pins the last address in
//           place and keeps re-issuing it, exactly like the normal hold
//           path does for every other address, until qk_data_valid and
//           qk_ready confirm it was actually consumed

module uram_pingpong_controller #(
    parameter ADDR_WIDTH = 10,
    parameter MATRIX_SIZE = 1024,
    // Per-token depth. NUM_TOKENS = MATRIX_SIZE / D_MODEL must equal
    // Module A's SIZE parameter: the whole point of D_MODEL below is to
    // reshape the flat, token-major MATRIX_SIZE buffer back into the
    // depth-major stream qk_array.sv actually expects (see its header
    // comment), and that only lines up dimensionally when the tile Module A
    // is built for (SIZE) matches how many tokens make up one package here.
    parameter D_MODEL = 64
)(
    input  logic clk,
    input  logic rst_n,

    // write side, comes from Module B, never stalls
    input  logic                  write_valid, // q_proj_v (Q,K,V valid at once)
    input  logic                  write_last,  // q_proj_last -- per-TOKEN, no
                                                 // longer used below (see note)
    output logic [ADDR_WIDTH:0]   waddr,       // +1 bit for bank select

    // read side, feeds Module A
    input  logic                  softmax_valid, // first result from Module C

    // qk_data_valid = q_uram_rvalid && k_uram_rvalid (this cycle's URAM read
    // result, if any); qk_ready = mod_a_input_arbiter's qk_ready (whether
    // that result is actually being accepted this cycle). uram_memory has
    // no backpressure of its own -- a result that isn't consumed the cycle
    // it appears is gone forever the moment the next read lands -- so these
    // are needed to safely pace the QK read stream against Module A's real
    // ready_o instead of free-running past it.
    input  logic                  qk_data_valid,
    input  logic                  qk_ready,

    output logic                  q_ren,
    output logic [ADDR_WIDTH:0]   q_raddr,
    output logic                  k_ren,
    output logic [ADDR_WIDTH:0]   k_raddr,
    output logic                  v_ren,
    output logic [ADDR_WIDTH:0]   v_raddr
);

    localparam int NUM_TOKENS = MATRIX_SIZE / D_MODEL;
    localparam int ROW_W      = (NUM_TOKENS <= 1) ? 1 : $clog2(NUM_TOKENS);
    localparam int DEPTH_W    = $clog2(D_MODEL);

    // Bank (Ping-Pong) Yönetimi
    logic wr_bank, rd_bank;
    logic [ADDR_WIDTH-1:0] wr_ptr, rd_ptr;

    // Kuyruktaki Hazır Paket Sayısı
    logic [1:0] pending_packages;

    // --- YAZMA MANTIĞI ---
    //
    // Previously reset (and counted a package as "written") on write_last
    // as well as wr_ptr reaching the end. write_last pulses once per TOKEN
    // (every D_MODEL scalars, from q_proj_last), not once per whole
    // MATRIX_SIZE sequence, so every single token was closing out a
    // "package" of its own: each new token's projection overwrote
    // addresses 0..D_MODEL-1 before the rest of the sequence had a chance
    // to accumulate, and Module A never actually saw more than the most
    // recently written token's worth of Q/K. A package is only genuinely
    // complete once the full MATRIX_SIZE = NUM_TOKENS*D_MODEL block has
    // been written, which wr_ptr == MATRIX_SIZE-1 alone already tracks.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_bank <= 1'b0;
            wr_ptr  <= '0;
        end else if (write_valid) begin
            if (wr_ptr == MATRIX_SIZE - 1) begin
                wr_ptr  <= '0;
                wr_bank <= ~wr_bank; // Banka değiştir (Ping -> Pong)
            end else begin
                wr_ptr <= wr_ptr + 1'b1;
            end
        end
    end
    assign waddr = {wr_bank, wr_ptr}; // 11-Bit Adres (MSB Banka, LSB Pointer)

    // --- PAKET TAKİBİ (Üretilen ama Modül A'nın henüz okumadığı paketler) ---
    logic pkg_written, pkg_read;
    assign pkg_written = write_valid && (wr_ptr == MATRIX_SIZE - 1);
    assign pkg_read    = (v_ren) && (rd_ptr == MATRIX_SIZE - 1); // V okuması bitince paket erimiştir

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pending_packages <= '0;
        end else begin
            if (pkg_written && !pkg_read)      pending_packages <= pending_packages + 1'b1;
            else if (!pkg_written && pkg_read) pending_packages <= pending_packages - 1'b1;
        end
    end

    // --- OKUMA MANTIĞI (Sadece okuma işlemi yönetilir, otoyol kesilmez) ---
    typedef enum logic [1:0] {ST_IDLE, ST_READ_QK, ST_WAIT_SOFTMAX, ST_READ_V} read_state_t;
    read_state_t rd_state;

    // Depth-major position within the current QK sweep. qk_array.sv wants,
    // for every depth step, all NUM_TOKENS row values back to back. The
    // URAM was written token-major (token 0's D_MODEL values, then token
    // 1's, ...), so walking it back out with the row index fastest and the
    // depth index slowest reproduces the depth-major order Module A needs
    // without moving any stored data around.
    logic [ROW_W-1:0]   qk_row;
    logic [DEPTH_W-1:0] qk_depth;

    // Set the cycle the very last (row=NUM_TOKENS-1, depth=D_MODEL-1)
    // address is issued. Every pair before this one relies on the *next*
    // pair's hold check to confirm it was actually consumed before the
    // sweep moves on; the last pair has no next iteration, so without this
    // flag rd_state left ST_READ_QK the instant its address went out,
    // before anything confirmed qk_ready was high when its result (one
    // cycle later) actually arrived. If it wasn't, that pair's data was
    // silently dropped and whatever qk_array captured for it was stale/X.
    logic final_addr_issued;

    // Hold at the current (row, depth) address instead of advancing
    // whenever this cycle's URAM result was valid but not accepted
    // downstream -- re-reading the same address is harmless and is the
    // only way to retry, since the memory itself won't hold a result for
    // us past the cycle it appears on.
    logic hold;
    assign hold = qk_data_valid && !qk_ready;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_state <= ST_IDLE;
            rd_bank  <= 1'b0;
            rd_ptr   <= '0;
            qk_row   <= '0;
            qk_depth <= '0;
            final_addr_issued <= 1'b0;
        end else begin
            case (rd_state)
                ST_IDLE: begin
                    rd_ptr   <= '0;
                    qk_row   <= '0;
                    qk_depth <= '0;
                    if (pending_packages > 0) rd_state <= ST_READ_QK; // Hazır paket varsa başla!
                end

                ST_READ_QK: begin
                    if (final_addr_issued) begin
                        // last address already went out; qk_row/qk_depth
                        // are still pinned at their final values (NOT reset
                        // yet) so q_ren keeps re-issuing that exact same
                        // address every cycle until hold confirms its
                        // result was actually accepted downstream.
                        if (!hold) begin
                            final_addr_issued <= 1'b0;
                            qk_row   <= '0;
                            qk_depth <= '0;
                            rd_state <= ST_WAIT_SOFTMAX;
                        end
                    end
                    else if (!hold) begin
                        if (qk_row == NUM_TOKENS - 1) begin
                            if (qk_depth == D_MODEL - 1) begin
                                // do not reset qk_row/qk_depth here -- pin
                                // them at (NUM_TOKENS-1, D_MODEL-1) until
                                // final_addr_issued's own retry/confirm
                                // above is done with them
                                final_addr_issued <= 1'b1;
                            end else begin
                                qk_row   <= '0;
                                qk_depth <= qk_depth + 1'b1;
                            end
                        end else begin
                            qk_row <= qk_row + 1'b1;
                        end
                    end
                    // else: hold qk_row/qk_depth, re-issue the same address
                end

                ST_WAIT_SOFTMAX: begin
                    rd_ptr <= '0;
                    if (softmax_valid) rd_state <= ST_READ_V; // Softmax damladığında V'yi oku
                end

                ST_READ_V: begin
                    if (rd_ptr == MATRIX_SIZE - 1) begin
                        rd_ptr   <= '0;
                        rd_bank  <= ~rd_bank; // İşlem bitti, diğer Ping/Pong bankasına geç
                        rd_state <= ST_IDLE;
                    end else begin
                        rd_ptr <= rd_ptr + 1'b1;
                    end
                end
            endcase
        end
    end

    assign q_ren   = (rd_state == ST_READ_QK);
    assign k_ren   = (rd_state == ST_READ_QK);
    assign v_ren   = (rd_state == ST_READ_V);

    logic [ADDR_WIDTH-1:0] qk_addr;
    assign qk_addr = qk_row * D_MODEL + qk_depth;

    assign q_raddr = {rd_bank, qk_addr};
    assign k_raddr = {rd_bank, qk_addr};
    assign v_raddr = {rd_bank, rd_ptr};

endmodule
