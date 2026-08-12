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