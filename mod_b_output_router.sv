module mod_b_output_router #(
    parameter DATA_WIDTH = 16,
    parameter MATRIX_SIZE = 1024 // Bir matristeki toplam eleman sayısı
)(
    input  logic clk,
    input  logic rst_n,

    // --- TEK Modül B'den Gelen Çıkış ---
    input  logic                  mod_b_valid,
    input  logic [DATA_WIDTH-1:0] mod_b_data,
    input  logic                  mod_b_tag,   // 1 Bit! (0: URAM'a, 1: FIFO'ya)
    output logic                  mod_b_ready,

    // --- Hedef 1, 2, 3: URAM'lar ---
    output logic                  q_uram_valid,
    output logic                  k_uram_valid,
    output logic                  v_uram_valid,
    output logic [DATA_WIDTH-1:0] uram_data,    
    input  logic                  q_uram_ready,
    input  logic                  k_uram_ready,
    input  logic                  v_uram_ready,

    // --- Hedef 4: TX FIFO ---
    output logic                  fifo_valid,
    output logic [DATA_WIDTH-1:0] fifo_data,
    input  logic                  fifo_ready
);

    // Veriyi saymak için sayaçlar (Sadece Tag 0 için çalışır)
    logic [15:0] element_counter; // 0'dan 1023'e kadar sayar
    logic [1:0]  matrix_state;    // 0: Q, 1: V, 2: K

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            element_counter <= '0;
            matrix_state    <= '0;
        end else begin
            // Eğer veri URAM içinse (tag==0) ve başarıyla iletildiyse (ready&&valid)
            if (mod_b_valid && mod_b_tag == 1'b0 && mod_b_ready) begin
                if (element_counter == MATRIX_SIZE - 1) begin
                    element_counter <= '0; // Paket bitti, sıfırla
                    
                    // Makası bir sonrakine çevir (Q -> V -> K döngüsü)
                    if (matrix_state == 2'd2)
                        matrix_state <= '0;
                    else
                        matrix_state <= matrix_state + 1'b1;
                end else begin
                    element_counter <= element_counter + 1'b1;
                end
            end
        end
    end

    // 1. Otoyol Dağıtımı
    assign uram_data = mod_b_data;
    assign fifo_data = mod_b_data;

    // 2. Tag 1 ise Doğrudan FIFO'ya
    assign fifo_valid = mod_b_valid && (mod_b_tag == 1'b1);

    // 3. Tag 0 ise Duruma Göre (matrix_state) URAM'lara Dağıt
    // KULLANICI İSTEĞİ: Sırayla Q (0) -> V (1) -> K (2)
    assign q_uram_valid = mod_b_valid && (mod_b_tag == 1'b0) && (matrix_state == 2'd0);
    assign v_uram_valid = mod_b_valid && (mod_b_tag == 1'b0) && (matrix_state == 2'd1);
    assign k_uram_valid = mod_b_valid && (mod_b_tag == 1'b0) && (matrix_state == 2'd2);

    // 4. Backpressure (Tıkanıklık) Yönlendirmesi
    always_comb begin
        if (mod_b_tag == 1'b1) begin
            mod_b_ready = fifo_ready;
        end else begin
            case (matrix_state)
                2'd0: mod_b_ready = q_uram_ready;
                2'd1: mod_b_ready = v_uram_ready;
                2'd2: mod_b_ready = k_uram_ready;
                default: mod_b_ready = 1'b0;
            endcase
        end
    end

endmodule