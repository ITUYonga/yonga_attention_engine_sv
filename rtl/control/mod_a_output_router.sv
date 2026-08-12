// FUNC_MOD_A_OUTPUT_ROUTER : Sends the shared array's output to softmax
// or to the output projection, depending on which pass produced it
//
//   Purpose:  The shared array is reused for two passes, Q times K
//             transpose feeding softmax next, and score times V feeding
//             the output projection next, and both come out of the same
//             physical output port. This module reads the tag bit that
//             rides along every result and steers it to the matching
//             destination, tag 0 goes to softmax, tag 1 goes to the
//             output (Wo) projection. Data and last are simply wired to
//             both destinations, only valid and ready are actually
//             switched by the tag.
//
//   parameters:
//           DATA_WIDTH:   bit width of the tagged bus, should be 17
//                         (16 bit value plus the 1 bit source tag)
//
//   inputs:
//           mod_a_valid:   the shared array has a result this cycle
//           mod_a_data:    that result, tag bit included
//           mod_a_tag:     0 routes to softmax, 1 routes to the output
//                          projection
//           mod_a_last:    this result is the last of its row/vector
//           c_ready:       softmax can accept a result this cycle
//           b_ready:       the output projection can accept a result
//                          this cycle
//   output:
//           mod_a_ready:   tells the shared array its result was
//                          accepted, mirrors whichever destination's
//                          ready the tag selected
//           c_valid, c_data, c_last:
//                          softmax side of the route
//           b_valid, b_data, b_last:
//                          output projection side of the route
//
//   notes:
//           only one of c_valid/b_valid is ever high at a time, decided
//           purely by mod_a_tag, there is no arbitration needed here
//           since the array itself only ever produces one result at once

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
