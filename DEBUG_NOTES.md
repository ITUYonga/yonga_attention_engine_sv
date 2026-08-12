# Uctan uca testbench (tb_top) - kaldigimiz yer

Bu dosya bilgisayar degistirirken kaldigimiz yeri not etmek icin. Guncel durum:

## GUNCELLEME 12.08.2026 - deadlock/veri kaybi bugleri cozuldu, pipeline tamamlaniyor

Asagidaki "SIRADAKI ADIM" bolumundeki plana gore devam edildi (done_o -> a_out_v
-> c_in_v -> c_v zincirini tracer'la takip ederek), sirayla 4 ayri gercek bug
bulunup duzeltildi:

1. **`rtl/control/dbuf.sv`** - `swap_buffers` (= `rx_we && internal_rx_last`)
   ile son elemanin kendi yazma islemi AYNI cycle'da tetikleniyordu, ama
   if/else oldugu icin swap_buffers dali kazaniyor, verinin gercek yazimi
   (`mem_ping[write_addr]<=rx_data`) hic calismiyordu. Her paketin SON
   elemani sonsuza dek X kaliyordu bellekte. Bu, sonraki her katmanda
   (uram_pingpong_controller, qk_array, qkv_proj ciktisi, sum_out) X olarak
   goruntulenen sorunun asil kaynagiydi. Fix: veri yazimi ile
   write_addr/write_to_pong bookkeeping'i artik ayri if'ler, birbirini
   engellemiyor.

2. **`rtl/control/uram_pingpong_controller.sv`** - ST_READ_QK, son (255.)
   QK adresini verir vermez, sonucunun gercekten kabul edildigini hic
   kontrol etmeden ST_WAIT_SOFTMAX'a geciyordu (diger butun ciftler icin
   "bir sonraki iterasyonun hold kontrolu" bu isi zimnen yapiyordu, sonuncu
   cift icin "bir sonraki" hic yoktu). `final_addr_issued` bayragiyla artik
   son adres, sonucu kabul edilene (`qk_data_valid && qk_ready`) kadar
   pinlenip tekrar tekrar okutuluyor.

3. **`rtl/control/mod_a_input_arbiter.sv`** - `c_ready` (softmax'in
   `m_axis_tready`'sine giden sinyal), `del_c_valid` (=`c_valid`'in 1-cycle
   gecikmeli kopyasi) dalinin icine gomulmustu. Softmax ilk kez `c_valid`
   verebilmesi icin `c_ready`'nin bir kez 1 olmasi gerekiyordu, ama
   `c_ready` ancak softmax DAHA ONCE `c_valid` vermisse 1 olabiliyordu -
   tavuk-yumurta deadlock. softmax INVERT/DIVIDE_NORMALIZE'a hicbir zaman
   ilk ciktisini uretmeden giriyor, sonsuza dek orada kaliyordu. Fix:
   `c_ready` artik dogrudan `mod_a_ready`'yi yansitiyor (gölge kaydin kendi
   yakalama kosuluyla ayni).

4. **`rtl/softmax/softmax.sv`** - `norm_multiplier` (bf16_mul, 2cc pipeline)
   cikisinda `scale_mask.sv`'de zaten bulunup duzeltilmis olan ayni sinif
   bug vardi: `fifo_rd_en` art arda cycle'larda tetiklenip iki mul_valid_o
   sonucunu ust uste bindirebiliyordu, tek register olan `m_axis_tvalid`
   ikincisini sessizce kaybediyordu. `scale_mask.sv`'nin `pipe_busy`
   deseni `norm_pipe_busy` olarak buraya da eklendi.

Bu 4 fix'ten sonra pipeline **uctan uca hic kilitlenmeden/veri kaybetmeden
tamamlaniyor**: `c_v` 16 elemanin tamami icin tetikleniyor (4x4 skor
matrisi), timeout yok.

**Acik kalan (sayisal dogruluk, plumbing degil):** softmax, 4x4 skor
matrisini TEK BUYUK satir olarak isliyor (16 eleman tek ACCUMULATE turu),
oysa her satirin (4 eleman) kendi bagimsiz softmax'i olmasi lazim. Kaynak
`mod_a_wrapper.sv`'nin `m_last` ciktisi (`row_cnt==SIZE-1 && col_cnt==SIZE-1`)
sadece matrisin en son elemaninda 1 oluyor, satir basina degil. Bunu
`col_cnt==SIZE-1` (satir basina) yapmayi denedik, softmax satir 0'i dogru
bitirdi ama sonra IDLE/ACCUMULATE'e donup satir 1'i kabul etmeyi beceremedi,
YENI bir deadlock'a yol acti (scale_mask'in tek eleman'lik pipe_busy slotu
hic bosalmadi). Kok nedeni bulunamadi, `m_last` eski (satir-siz, sadece
matris-sonu) haline GERI ALINDI - pipeline'in tamamlanmasini
(deadlock'suz calismasini) korumak icin. Yani su an dogru CALISIYOR ama
softmax'in urettigi sayisal degerler golden model ile TAM ESLESMIYOR
(satir toplami ~4x buyuk cikiyor). Sonraki adim: once neden per-row tlast
ikinci deadlock'a yol aciyor onu bulmak, sonra tekrar uygulamak.

Ilgili commit: `1a9b18c` ("uctan uca debug: 4 gercek RTL bug'i bulundu ve
duzeltildi"), branch `integration/attention-merge`.

## Ne yapiliyor

`tb/tb_top.sv`: `rtl/top.sv` (attention_engine_top) icin uctan uca bir testbench.
Kapsam: AXI-Stream token girisi -> QKV projeksiyon -> RoPE -> ping-pong URAM ->
paylasilan sistolik dizi (Q.K^T gecisi) -> scale -> softmax cikisi. score.V
gecisi ve cikis projeksiyonu **bilerek kapsam disi** (asagida neden bkz).

Senaryo: tam olarak 4 token (mod_a_wrapper'in top.sv'de sabit SIZE=4 oldugu
icin baska sayida olamaz), D_MODEL=64 (RoPE'un rope_sin_rom.mem/rope_cos_rom.mem
dosyalari bu deger icin uretilmis, degistirilemez). Token t, standart taban
vektoru e_t (t. boyutta 1.0, gerisi 0). Wq=identity, Wk=2*identity, boylece
projeksiyon sonrasi Q=token, K=2*token oluyor, elle dogrulanmasi kolay.

Altin model (`tb/bf16_bitexact_utils.svh` + `tb_top.sv` icindeki
`build_golden_model()`): RTL'deki bf16_mul/bf16_add/exp_lut/recip_lut'un
bit-birebir portu, gercek ROM/LUT dosyalarini okuyarak hesapliyor. Bu kisim
dogru calisiyor, hesaplanan degerler mantikli (kosegen ~2.0, kosegen disi
tam 0.0, softmax agirliklari makul).

## Bulunan ve duzeltilen gercek buglar (bu RTL hic uctan uca test edilmemisti)

`rtl/control/uram_pingpong_controller.sv` icinde ucu bug vardi:

1. Yazma tarafi "paket tamam" sayacini her TOKEN'da sifirliyordu
   (`write_last`, her 64 elemanda bir), tum MATRIX_SIZE=256 elemanlik
   diziyi degil. Sadece en son token hayatta kaliyordu.
2. Okuma tarafi URAM'dan duz (token-major) sirada okuyup dogrudan Module
   A'ya gonderiyordu, ama qk_array.sv'nin header yorumunda acikca yazdigi
   gibi depth-major sira bekliyor. Hic yeniden siralama yoktu.
3. Okuma tarafinin hic backpressure'i yoktu, Module A hazir olmasa bile her
   cycle yeni okuma yapip bir onceki sonucu kaybediyordu.

Uc uzu de duzeltildi (commit: "uram_pingpong_controller: yazma tarafi...").
`rtl/top.sv`'ye de bunun icin `qk_data_valid`/`qk_ready` portlari eklendi.

**Onemli**: score.V gecisinde de AYRI bir sira uyusmazligi var (softmax
skorlari satir-major sirada cikiyor ama Module A'nin ikinci gecisi farkli
bir sira bekliyor, V okumalarinin softmax'in cikis hizina gore
eslenmesi de yok). Bu HENUZ duzeltilmedi, bilerek. tb_top.sv sadece
Q.K^T -> softmax yarisini test ediyor.

## Simulasyonun mevcut durumu (Vivado, en son calistirilan hal)

Derleme/elaborasyon TEMIZ (Vivado projesindeki eski "imports" kopyalari
temizlendikten sonra, asagiya bak). Simulasyon calisiyor ama **softmax hic
cikti uretmiyor**: `c_v` 5ms boyunca hic 1 olmuyor, testbench 0/16 attention
weight ile timeout'a giriyor.

`get_value` ile kontrol edilen son durum (5ms'de):
```
pos_cnt=4                    -> tum 4 token islendi (giris tarafi)
wr_ptr=0                     -> 256 eleman yazildi, MATRIX_SIZE'a ulasip sifira sardi (basarili!)
pending_packages=1           -> paket tamamlandi (basarili!)
rd_state=ST_WAIT_SOFTMAX     -> okuma tarafi TUM 256 QK ciftini basariyla Module A'ya besledi (depth-major fix calisiyor!)
qk_valid=0, qk_ready=0       -> QK okuma fazi bitmis (beklenen)
a_in_v=0, a_in_r=1           -> Module A su an bos/hazir
```

Yani URAM<->Module A arasindaki 3 bug duzeltmesi ISE YARADI, 256 QK cifti
basariyla Module A'ya ulasti. Kullanici dalga formunda `a_out_v`'nin en az
bir kere 1'e ciktigini gozle dogruladi (zoom yaparak). Ama `c_v` (softmax
nihai ciktisi) hic 1 olmuyor.

## SIRADAKI ADIM (henuz calistirilmadi, kaldigimiz yer burasi)

`tb_top.sv`'ye tam zamanli bir debug tracer eklendi (en son commit):
```systemverilog
initial begin
    forever begin
        @(posedge clk);
        if (dut.u_mod_a_wrap.u_qk_array.done_o) $display("[%0t] DEBUG qk_array.done_o pulsed", $time);
        if (dut.a_out_v) $display("[%0t] DEBUG a_out_v=1 a_out_d=%h a_out_last=%b a_out_r=%b", $time, dut.a_out_d, dut.a_out_last, dut.a_out_r);
        if (dut.c_in_v) $display("[%0t] DEBUG c_in_v=1 c_in_d=%h c_in_last=%b c_in_r=%b", $time, dut.c_in_d, dut.c_in_last, dut.c_in_r);
        if (dut.c_v) $display("[%0t] DEBUG c_v=1 c_d=%h c_r=%b", $time, dut.c_d, dut.c_r);
    end
end
```

Yapilacak: Vivado'da `launch_simulation` (dosya degisti, otomatik yeniden
derler) sonra `run -all`, cikan konsol logunu incele:

- `done_o` hic yazdirilmiyorsa -> Module A kendi icinde islemi hic
  bitirmiyor (qk_array.sv'nin done_o mantiginda bir sorun, veya 256.
  cift dispatch edilmiyor).
- `done_o` cikiyor ama `a_out_v` cikmiyorsa -> mod_a_wrapper'in drain
  mantiginda sorun (out_buffer<=sum_out capture, drain_active tetiklenmesi).
- `a_out_v` cikiyor ama `c_in_v` hic cikmiyorsa -> mod_a_output_router'da
  sorun (tag yanlis okunuyor olabilir, c_ready hep 0 kalip mod_a_ready'yi
  dusuruyor olabilir, vs).
- `c_in_v` cikiyor ama `c_v` hic cikmiyorsa -> scale_mask_softmax_wrapper
  veya softmax.sv'nin kendi icinde bir sorun var (belki en_scale_mask=0 ile
  ilgili, belki softmax'in ACCUMULATE/DIVIDE_NORMALIZE state'lerinde bir
  starvation).

Buradan devam et.

## Vivado proje kurulumuyla ilgili notlar (tekrar karsilasilirsa)

- Proje: `C:/Users/can07/attention_accelerator`, dosya: `attention_accelerator.xpr`.
- Git branch: `integration/attention-merge` (main degil).
- Projede baslangicta HER dosyanin bir de `.srcs/sources_1/imports/rtl/...`
  altinda ESKI/DONMUS kopyasi vardi (muhtemelen proje ilk kurulurken "copy
  sources" ile). Bunlar temizlendi (`remove_files` ile), ama ONEMLI: bazi
  dosyalarin (bf16_add, bf16_comb, bf16_convert, bf16_mul, qk_pe, gqa_mapper,
  projection_block, qkv_proj, rope, row_fifo, scale_mask_softmax_wrapper)
  SADECE o imports kopyasi vardi, gercek path'e referans yoktu -- onlari
  `add_files -fileset sources_1` ile gercek `rtl/...` path'inden tekrar
  eklemek gerekti. Eger "module not found" / "port not found" / "parameter
  not found" gibi elaborasyon hatalari tekrar cikarsa, once
  `get_files -all *.sv` ile "imports" gecen bir satir var mi kontrol et.
- `rope_sin_rom.mem`, `rope_cos_rom.mem`, `exp_table.hex`, `recip_table.hex`
  Vivado'nun xsim calisma klasorune (`<proje>.sim/sim_1/behav/xsim`)
  otomatik export edilirken alt klasor yapisi (`rtl/projection/`,
  `rtl/softmax/`) DUSUYOR, ama RTL'nin varsayilan parametre path'i hala o
  alt klasoru istiyor. xsim klasoru her silinip yeniden olusturuldugunda
  su Tcl'yi tekrar calistirmak gerekiyor:
  ```tcl
  set xdir {C:/Users/can07/attention_accelerator/attention_accelerator.sim/sim_1/behav/xsim}
  file mkdir $xdir/rtl/projection
  file mkdir $xdir/rtl/softmax
  file copy -force {C:/Users/can07/OneDrive/Masaüstü/fpt26-/rtl/projection/rope_sin_rom.mem} $xdir/rtl/projection/rope_sin_rom.mem
  file copy -force {C:/Users/can07/OneDrive/Masaüstü/fpt26-/rtl/projection/rope_cos_rom.mem} $xdir/rtl/projection/rope_cos_rom.mem
  file copy -force {C:/Users/can07/OneDrive/Masaüstü/fpt26-/rtl/softmax/exp_table.hex} $xdir/rtl/softmax/exp_table.hex
  file copy -force {C:/Users/can07/OneDrive/Masaüstü/fpt26-/rtl/softmax/recip_table.hex} $xdir/rtl/softmax/recip_table.hex
  ```
- Bir token'in projeksiyonu (64x64 MAC dongusu) gercekten yavas, ~250-450us
  suruyor. 4 token + Module A + softmax icin en az birkac ms simulasyon
  suresi gerekiyor, testbench'teki watchdog 5ms'de ($time) $finish cagiriyor
  (`tb_top.sv` sonundaki `initial begin #5000000; ... end`), gerekirse
  bunu buyut.
