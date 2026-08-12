# Uctan uca testbench (tb_top) - kaldigimiz yer

Bu dosya bilgisayar degistirirken kaldigimiz yeri not etmek icin. Guncel durum:

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
