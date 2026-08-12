`timescale 1ns / 1ps

// FUNC_URAM_MEMORY : One synchronous read/write memory bank, mapped to
// UltraRAM style block memory
//
//   Purpose:  A plain single bank memory used to hold one of the Q, K, or
//             V arrays. Write address comes directly from whatever
//             controller is driving this bank (uram_pingpong_controller
//             for Q/K/V storage), rather than being generated internally,
//             so several banks can share one controller's addressing
//             scheme. Reads are synchronous with one cycle of latency,
//             matching how real UltraRAM blocks behave, and rvalid tells
//             the reader exactly which cycle rdata is actually valid.
//
//   parameters:
//           DATA_WIDTH:   bit width of one stored element
//           ADDR_WIDTH:   address width, memory depth is DEPTH regardless
//                         of how many address bits this allows
//           DEPTH:        number of storable elements
//
//   inputs:
//           clk, rst_n:   clock and active low reset
//           wr_en:        wr_data is valid this cycle, write it to
//                         wr_addr
//           wr_addr:      address to write to this cycle
//           wr_data:      one element to store
//           rd_en:        read rd_addr this cycle, result appears next
//                         cycle
//           rd_addr:      address to read this cycle
//   output:
//           ready:        always 1 here, this bank never applies its own
//                         backpressure, whoever writes to it is expected
//                         to already know it has room
//           rdata:        data read from the address rd_addr held one
//                         cycle ago
//           rvalid:       rdata is valid this cycle, mirrors last
//                         cycle's rd_en
//
//   notes:
//           the ram_style ultra attribute is a Vivado synthesis hint,
//           asking the tool to map the storage array onto UltraRAM
//           primitives instead of ordinary Block RAM or distributed LUT
//           RAM

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
    input  logic [ADDR_WIDTH-1:0] wr_addr,
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

    // ==========================================
    // YAZMA İŞLEMİ (adres dışarıdan geliyor - uram_pingpong_controller
    // bank+pointer'ı zaten hesaplıyor, burada ikinci bir sayaca gerek yok.
    // Önceki sürümde bu iç sayaçla üretiliyordu ve wr_addr diye bir port
    // hiç yoktu; top.sv'nin bağlamaya çalıştığı adres gidecek yer bulamıyordu)
    // ==========================================
    always_ff @(posedge clk) begin
        if (wr_en) begin
            mem[wr_addr] <= wr_data;
        end
    end

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