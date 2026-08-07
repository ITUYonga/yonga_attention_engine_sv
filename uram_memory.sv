`timescale 1ns / 1ps

module uram_memory #(
    parameter DATA_WIDTH = 16,
    parameter ADDR_WIDTH = 10,   // 1024 eleman için
    parameter DEPTH = 1024       // Bellek derinliği
)(
    input  logic clk,
    input  logic rst_n,

    // ==========================================
    // 1. YAZMA ARAYÜZÜ (Modül B'den Gelir - Streaming)
    // ==========================================
    input  logic                  wr_en,
    input  logic [DATA_WIDTH-1:0] wr_data,
    output logic                  ready,     // Modül B'ye "Hazırım" sinyali

    // ==========================================
    // 2. OKUMA ARAYÜZÜ (Modül A Hakemi/Çekirdeği'nden Gelir)
    // ==========================================
    input  logic                  rd_en,
    input  logic [ADDR_WIDTH-1:0] rd_addr,
    output logic [DATA_WIDTH-1:0] rdata,
    output logic                  rvalid     // Okuma işleminin 1 cycle gecikmesi (latency) bittiğinde 1 olur
);

    // VIVADO SİHRİ: Bu etiketi gören Vivado, diziyi doğrudan çipteki URAM bloklarına haritalar!
    (* ram_style = "ultra" *) 
    logic [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    // Yazma işlemi için iç sayaç (Modül B adres göndermediği için biz sayıyoruz)
    logic [ADDR_WIDTH-1:0] wr_addr;
    logic is_full;

    // ==========================================
    // YAZMA İŞLEMİ VE SAYAÇ KONTROLÜ
    // ==========================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_addr <= '0;
        end else begin
            if (wr_en && !is_full) begin
                mem[wr_addr] <= wr_data; // Belleğe yaz
                
                // Güvenlik: Adres sınırına ulaştıysa başa sar (veya bekle)
                if (wr_addr == DEPTH - 1)
                    wr_addr <= '0;
                else
                    wr_addr <= wr_addr + 1'b1;
            end
        end
    end

    // URAM kapasitesi dolmadıysa Modül B'den veri almaya her zaman hazırdır
    assign is_full = (wr_addr == DEPTH - 1 && wr_en); 
    assign ready   = 1'b1; // (Dataflow tasarımımızda URAM hep veri kabul edecek kadar büyük tasarlanır)

    // ==========================================
    // OKUMA İŞLEMİ (Senkron - 1 Clock Gecikmeli)
    // ==========================================
    // UltraRAM'ler senkron çalışır. Adres verildikten 1 saat vuruşu sonra veri gelir.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rdata  <= '0;
            rvalid <= 1'b0;
        end else begin
            if (rd_en) begin
                rdata  <= mem[rd_addr]; // İstenen adresi okuyup çıkışa yaz
                rvalid <= 1'b1;         // "Veri hazır" bayrağını kaldır
            end else begin
                rvalid <= 1'b0;
            end
        end
    end

endmodule