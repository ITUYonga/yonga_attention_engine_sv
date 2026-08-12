// FUNC_MOD_A_INPUT_ARBITER : Picks what feeds the shared systolic array
// next, softmax's score times V pass or a fresh Q times K pair
//
//   Purpose:  The shared array (qk_array/mod_a_wrapper) is reused for two
//             different passes, so this module decides which one gets to
//             go in on a given cycle. Softmax's output always takes
//             priority when it has something ready, since letting a
//             finished row sit around only delays the output projection
//             further down. Otherwise, whenever both URAM read ports for
//             Q and K have valid data, that pair goes in tagged 0. A one
//             cycle shadow register (del_c_data/del_c_valid) delays
//             softmax's score by exactly one cycle so it lines up with
//             v_rdata, which itself has one cycle of URAM read latency,
//             so both halves of a score times V pair really do refer to
//             the same row element.
//
//   parameters:
//           DATA_WIDTH:   bit width of one bf16 element on this bus,
//                         should be 17 wherever this is instantiated
//                         (16 bit value plus the 1 bit source tag)
//
//   inputs:
//           clk, rst_n:    clock and active low reset
//           c_valid:       softmax has a normalized weight ready
//           c_data:        that weight, tag bit included
//           v_rdata:       V element read out of URAM this cycle
//           qk_valid:      both Q and K URAM reads are valid this cycle
//           q_data, k_data: the Q and K elements themselves
//           mod_a_ready:   the shared array can accept a pair this cycle
//   output:
//           c_ready:       tells softmax its weight was accepted into
//                          the shadow register this cycle
//           qk_ready:      tells the URAM read side its Q/K pair was
//                          accepted this cycle
//           mod_a_valid:   a pair is being presented to the array
//           mod_a_data_l, mod_a_data_t:
//                          the two operands being presented, q_in and
//                          k_in respectively on the array's ports
//           mod_a_tag:     0 for a Q times K pair, 1 for a score times V
//                          pair, mirrors whichever operand's own tag bit
//                          actually reaches the array
//
//   notes:
//           v_rdata's own tag bit is always 0 because it came from Can's
//           qkv_proj/rope output, it gets forced to 1 here so it matches
//           del_c_data's tag before both reach the array, since qk_pe
//           asserts its two operands must carry the same tag
//
//           found and fixed a real deadlock here, c_ready used to be
//           driven only inside the del_c_valid branch, but del_c_valid is
//           itself just a one cycle delayed copy of c_valid. That meant
//           softmax could only be told ready the cycle after it had
//           already produced a c_valid, which it could never do without
//           c_ready going high first, a circular dependency that left
//           softmax stuck waiting forever for its very first result.
//           c_ready now simply mirrors mod_a_ready directly, matching the
//           shadow register's own capture condition, independent of
//           whether del_c_valid happens to be selected for mod_a_valid
//           on that same cycle

module mod_a_input_arbiter #(parameter DATA_WIDTH = 16) (
    input  logic clk, rst_n,
    // Softmax'ten (Modül C) Gelenler
    input  logic                  c_valid,
    input  logic [DATA_WIDTH-1:0] c_data,
    output logic                  c_ready,
    // URAM_V'den gelen veri
    input  logic [DATA_WIDTH-1:0] v_rdata, 
    // QxK URAM'lardan Gelenler
    input  logic                  qk_valid,
    input  logic [DATA_WIDTH-1:0] q_data, k_data,
    output logic                  qk_ready,
    // Tek Modül A'ya Giden
    output logic                  mod_a_valid,
    output logic [DATA_WIDTH-1:0] mod_a_data_l, mod_a_data_t,
    output logic                  mod_a_tag,
    input  logic                  mod_a_ready
);
    // V Matrisi Gecikme Senkronizasyonu (Shadow Register)
    logic [DATA_WIDTH-1:0] del_c_data; logic del_c_valid;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin del_c_valid <= 1'b0; del_c_data <= '0; end 
        else if (mod_a_ready) begin del_c_valid <= c_valid; del_c_data <= c_data; end
    end

    // c_ready must NOT be nested inside "if (del_c_valid)": del_c_valid is
    // itself just a one-cycle-delayed copy of c_valid (see shadow register
    // above), so gating c_ready behind it means softmax can only ever be
    // told "ready" the cycle after it has already produced a c_valid --
    // which it can never do without c_ready going high first. That
    // chicken-and-egg deadlock is exactly what left softmax stuck forever
    // in DIVIDE_NORMALIZE with fifo_rd_en never pulsing. The shadow
    // register's own capture condition (mod_a_ready) is the real
    // "can I accept a new c_valid this cycle" signal; c_ready just needs to
    // mirror that directly, independent of whether a previously-shadowed
    // value happens to be selected for mod_a_valid this same cycle.
    assign c_ready = mod_a_ready;

    always_comb begin
        qk_ready = 1'b0; mod_a_valid = 1'b0; mod_a_tag = 1'b0;
        mod_a_data_l = '0; mod_a_data_t = '0;

        if (del_c_valid) begin // Öncelik Softmax (Tag 1)
            // NOT: v_rdata'nin kendi tag biti (bit DATA_WIDTH-1) Can'in
            // qkv_proj/rope ciktisindan geldigi icin hep 0. qk_pe.sv'nin
            // assert(q_in[16]==k_in[16]) sartini gecebilmesi icin burada
            // 1'e zorlaniyor (deger bitlerine dokunmuyor). Takimla teyit
            // edilmesi gereken bir varsayim, bkz. entegrasyon notlari.
            mod_a_valid = 1'b1; mod_a_data_l = del_c_data; mod_a_data_t = {1'b1, v_rdata[DATA_WIDTH-2:0]}; mod_a_tag = 1'b1;
        end else if (qk_valid) begin // Sonra QxK (Tag 0)
            mod_a_valid = 1'b1; mod_a_data_l = q_data; mod_a_data_t = k_data; mod_a_tag = 1'b0;
            qk_ready = mod_a_ready;
        end
    end
endmodule