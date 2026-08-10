module mod_a_output_router #(
    parameter DATA_WIDTH = 17
)(
    // Modül A'dan gelen girişler
    input  logic                  mod_a_valid,
    input  logic [DATA_WIDTH-1:0] mod_a_data,
    input  logic                  mod_a_tag,
    input  logic                  mod_a_last,  // <--- YENİ EKLENDİ
    output logic                  mod_a_ready,

    // Hedef C: Softmax
    output logic                  c_valid,
    output logic [DATA_WIDTH-1:0] c_data,
    output logic                  c_last,      // <--- YENİ EKLENDİ
    input  logic                  c_ready,

    // Hedef B: W0 Projeksiyonu
    output logic                  b_valid,
    output logic [DATA_WIDTH-1:0] b_data,
    output logic                  b_last,      // <--- YENİ EKLENDİ
    input  logic                  b_ready
);

    // Veri ve Last yolları fiziksel olarak iki tarafa da bağlıdır
    assign c_data = mod_a_data;
    assign b_data = mod_a_data;
    assign c_last = mod_a_last;
    assign b_last = mod_a_last;

    // Yönlendirme (Valid ve Ready) mantığı Tag'e göre belirlenir
    always_comb begin
        c_valid = 1'b0;
        b_valid = 1'b0;
        mod_a_ready = 1'b0;

        if (mod_a_tag == 1'b0) begin
            c_valid     = mod_a_valid;
            mod_a_ready = c_ready;
        end else begin
            b_valid     = mod_a_valid;
            mod_a_ready = b_ready;
        end
    end
endmodule
