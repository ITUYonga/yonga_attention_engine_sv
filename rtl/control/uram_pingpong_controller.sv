`timescale 1ns / 1ps

module uram_pingpong_controller #(
    parameter ADDR_WIDTH = 10,
    parameter MATRIX_SIZE = 1024
)(
    input  logic clk,
    input  logic rst_n,

    // ==========================================
    // YAZMA HATTI (Modül B'den gelir, hiç durmaz)
    // ==========================================
    input  logic                  write_valid, // q_proj_v (Q,K,V aynı anda geçerlidir)
    input  logic                  write_last,  // q_proj_last
    output logic [ADDR_WIDTH:0]   waddr,       // +1 bit Bank seçimi için

    // ==========================================
    // OKUMA HATTI (Modül A'yı besler)
    // ==========================================
    input  logic                  softmax_valid, // Modül C'den gelen ilk sonuç

    output logic                  q_ren,
    output logic [ADDR_WIDTH:0]   q_raddr,
    output logic                  k_ren,
    output logic [ADDR_WIDTH:0]   k_raddr,
    output logic                  v_ren,
    output logic [ADDR_WIDTH:0]   v_raddr
);

    // Bank (Ping-Pong) Yönetimi
    logic wr_bank, rd_bank;
    logic [ADDR_WIDTH-1:0] wr_ptr, rd_ptr;
    
    // Kuyruktaki Hazır Paket Sayısı
    logic [1:0] pending_packages;

    // --- YAZMA MANTIĞI ---
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_bank <= 1'b0;
            wr_ptr  <= '0;
        end else if (write_valid) begin
            if (write_last || wr_ptr == MATRIX_SIZE - 1) begin
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
    assign pkg_written = write_valid && (write_last || wr_ptr == MATRIX_SIZE - 1);
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

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_state <= ST_IDLE;
            rd_bank  <= 1'b0;
            rd_ptr   <= '0;
        end else begin
            case (rd_state)
                ST_IDLE: begin
                    rd_ptr <= '0;
                    if (pending_packages > 0) rd_state <= ST_READ_QK; // Hazır paket varsa başla!
                end
                
                ST_READ_QK: begin
                    if (rd_ptr == MATRIX_SIZE - 1) begin
                        rd_ptr   <= '0;
                        rd_state <= ST_WAIT_SOFTMAX;
                    end else begin
                        rd_ptr <= rd_ptr + 1'b1;
                    end
                end

                ST_WAIT_SOFTMAX: begin
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
    
    assign q_raddr = {rd_bank, rd_ptr};
    assign k_raddr = {rd_bank, rd_ptr};
    assign v_raddr = {rd_bank, rd_ptr};

endmodule
