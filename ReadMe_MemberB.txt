	YONGA ATTENTION ENGINE İÇİN QKV PROJECTION, ROPE VE GQA MAPPER MODÜLLERİ
	CAN ERTÜRK - 21.07.2026
	FPT26 YONGA ATTENTION ENGINE - İTÜ YONGA TAKIMI - MEMBER B

	--------------------------------------------------------------------------------------------------------------------------------------

	Amaç: Bu dosya YAE attention hızlandırıcısı içinde benim sorumlu olduğum üç RTL modülünü anlatıyor, qkv_proj.sv,
	      rope.sv ve gqa_mapper.sv. Bu üç modül birlikte ham bir token embedding'ini alıp pipeline'ın ihtiyacı olan
	      Q, K ve V vektörlerine çeviriyor, Q ve K'ya pozisyon bilgisini işliyor, ve çok sayıdaki Q head'inin daha az
	      sayıdaki K/V head'inden hangisini okuması gerektiğini söylüyor.

	      Bu modüllerin gerçek çarpma toplama işlemleri için Belinay'ın bf16_add ve bf16_mul modüllerine ihtiyacı var,
	      çıkışları da Belinay'ın qk_array'ine (QK transpoze için), sonunda Hasan'ın scale_mask/softmax'ına ve
	      Taha'nın kv_cache'ine gidiyor. Bunların hiçbiri henüz bağlanmadı, bu doküman sadece kendi üç dosyamı anlatıyor.

	      Her şeyi önce basit bir state machine olarak yazıyorum, tek bir çarpıcı ve tek bir toplayıcı tekrar tekrar
	      kullanılıyor, hızlı paralel bir tasarım değil. Şu anki hedef doğru ve anlaşılır bir ilk versiyon, hız konusu
	      takım gerçekten gerekli olduğunu söyledikten sonra gelecek.

	--------------------------------------------------------------------------------------------------------------------------------------

	MİMARİ GENEL BAKIŞ

		Kısa Açıklama : Üç modülümün tüm YAE pipeline'ı içinde, baştan sona, tek bir token üzerinden nereye
				oturduğu.

			1. Bir token embedding vektörü Taha'nın axi_stream_if'inden içeri giriyor ve dbuf'unda tutuluyor.

			2. Benim qkv_proj.sv'im 4 kere instantiate ediliyor, Wq, Wk, Wv ve çıkış tarafı için Wo. Wq, Wk, Wv
			   instance'ları aynı token vektörünü dbuf'tan okuyup bir Q vektörü, bir K vektörü ve bir V vektörü
			   üretiyor.

			3. Benim rope.sv'im iki kere instantiate ediliyor, biri Q vektörüyle biri K vektörüyle beslenerek,
			   token'in pozisyonuyla (pos_i) birlikte. V hiç rope'tan geçmiyor, olduğu gibi bir sonraki aşamaya
			   gidiyor.

			4. Benim gqa_mapper.sv'im vektörlere hiç dokunmuyor, sadece şu anki Q head index'ini alıp o Q
			   head'inin Taha'nın kv_cache'inden hangi KV head'ini okuması gerektiğini söylüyor. O head'in K ve
			   V'si doğrudan benim modüllerimden değil kv_cache'ten geliyor, benim modüllerim sadece kv_cache'e
			   yazılacak K ve V'yi başlangıçta üretiyor.

			5. Buradan sonra rotasyona uğramış Q, ve gqa_mapper index'i kullanılarak kv_cache'ten okunan K/V,
			   Belinay'ın qk_array'ine Q çarpı K transpoze adımı için gidiyor, sonra Hasan'ın scale_mask.sv ve
			   softmax.sv'ine, sonra tekrar Belinay'ın qk_array'ine (aynı fiziksel array tekrar kullanılarak)
			   score çarpı V adımı için gidiyor.

			6. O head için attention çıktısı tekrar benim qkv_proj.sv'ime, bu sefer Wo ağırlık matrisiyle
			   yüklenmiş haliyle, geri gidip son çıkış vektörünü üretiyor, o da Taha'nın dbuf/axi_stream_if'inden
			   dışarı çıkıyor.

			7. Bütün bunlar Taha'nın top_fsm.sv'i tarafından sırayla çalıştırılıyor, yani benim modüllerimin ne
			   zaman çalışacağına asıl karar veren o, hiçbir modülüm kendi kendine başlamıyor, sadece x_ready_o'yu
			   yüksek tutup birinin x_valid_i vermesini bekliyorlar.

		Not: 5. ve 6. adımlar kendi dosyalarımın dışında, burada sadece bütün akış anlaşılsın diye listeledim,
		     onları ben yazıyorum diye değil.

	--------------------------------------------------------------------------------------------------------------------------------------

	"qkv_proj.sv"

	Fonksiyon Tanımı:

		qkv_proj.sv bir girdi vektörünü bir ağırlık matrisiyle çarpıp çıkış vektörüne çeviriyor, yani y = W*x
		işlemini tek tek çarpma toplama yaparak yapıyor. Aynı modül en üst seviyede dört kere instantiate
		ediliyor, biri Q biri K biri V projeksiyonu için, bir tanesi de attention'ın sonundaki çıkış
		projeksiyonu için, her instance'a farklı bir ağırlık matrisi yükleniyor.

		PARAMETRELER:

			DATA_WIDTH		- bir bf16 sayısının bit genişliği, 16
			D_MODEL			- girdi vektörünün uzunluğu
			D_OUT			- çıkış vektörünün uzunluğu, GQA'da K/V için D_MODEL'den küçük olabilir

		PORTLAR:

			w_data_i, w_addr_i, w_we_i	- kullanmadan önce ağırlık matrisini yüklemek için düz yazma portu
			x_data_i, x_valid_i, x_last_i	- girdi vektörü, clock başına bir eleman akıyor
			y_data_o, y_valid_o, y_last_o	- çıkış vektörü, clock başına bir eleman akıyor
			busy_o				- modül idle değilken yüksek, sadece waveform debug için

		STATE MACHINE ADIMLARI:

			ADIM 1 : ST_LOAD
			Kısa Açıklama : x_ready_o yüksekken girdi vektörünü clock başına bir eleman olarak x_mem'e
					 alıyor. x_last_i geldiğinde modül ST_COMPUTE'a geçiyor.

			ADIM 2 : ST_COMPUTE
			Kısa Açıklama : Çarpma toplama döngüsünü çalıştırıyor. İki iç içe sayaç var, out_idx_q hangi
					 çıkış elemanını oluşturduğumuzu, mac_idx_q hangi girdi elemanıyla çarptığımızı
					 tutuyor. Tek bir bf16_mul ve tek bir bf16_add her terim için tekrar tekrar
					 kullanılıyor, bu adım her çıkış elemanı için D_MODEL clock cycle sürüyor.

			ADIM 3 : ST_DRAIN
			Kısa Açıklama : y_ready_i yüksekken bitmiş y_mem içeriğini clock başına bir eleman dışarı
					 akıtıyor, y_last_o son elemanı işaretliyor.

		Not: weight_mem düz bir array index ile okunuyor, bu asenkron bir okuma, Taha'ya dbuf.sv için
		     söylediğim aynı endişe burada da geçerli. İlk versiyon için böyle bırakıldı, önce doğruluk.

		Not: bu tasarım bf16_mul ve bf16_add'in şu anki gibi tamamen combinational (0 cycle latency) kalacağını
		     varsayıyor. Belinay ileride pipeline eklerse ST_COMPUTE'a ekstra bekleme cycle'ı eklemek gerekecek.

	--------------------------------------------------------------------------------------------------------------------------------------

	"rope.sv"

	Fonksiyon Tanımı:

		rope.sv bir vektörün eleman çiftlerini token pozisyonuna göre indexlenmiş bir rom'dan çekilen sin ve
		cos değerleriyle döndürerek pozisyon bilgisini işliyor. Bu modül sadece Q ve K'ya dokunuyor, V hiçbir
		zaman bu modülden geçmiyor. En üst seviyede iki kere instantiate ediliyor, biri Q'ya biri K'ya
		bağlanıyor, iki instance da aynı rom içeriğini paylaşıyor.

		Bir (x1, x2) çifti için theta açısında rotasyon şöyle:
			y1 = x1*cos(theta) - x2*sin(theta)
			y2 = x1*sin(theta) + x2*cos(theta)

		PARAMETRELER:

			DATA_WIDTH		- bir bf16 sayısının bit genişliği, 16
			D_MODEL			- vektörün uzunluğu, çift sayı olmalı
			MAX_POS			- rom'un oluşturulduğu en büyük token pozisyonu

		PORTLAR:

			pos_i				- şu anki token'in pozisyonu, vektörün ilk elemanında bir kere latch'leniyor
			x_data_i, x_valid_i, x_last_i	- girdi vektörü, clock başına bir eleman akıyor
			y_data_o, y_valid_o, y_last_o	- dönmüş çıkış vektörü, clock başına bir eleman akıyor

		STATE MACHINE ADIMLARI:

			ADIM 1 : ST_LOAD
			Kısa Açıklama : qkv_proj ile aynı mantık, önce bütün girdi vektörünü x_mem'e topluyor, ayrıca
					 vektörün ilk elemanında pos_i'yi pos_q'ya latch'liyor.

			ADIM 2 : ST_COMPUTE
			Kısa Açıklama : Bir seferde bir çift üzerinde çalışıyor (pair_idx_q), her çiftin içinde de dört
					 mikro adım (mstep_q 0'dan 3'e) çalıştırıyor, çünkü rotasyon dört çarpma iki
					 toplama gerektiriyor ama elimizde sadece tek bir paylaşılan çarpıcı ve tek bir
					 paylaşılan toplayıcı var.

						mstep 0 : x1 * cos			-> tmp1'e yazılır
						mstep 1 : x2 * sin, sonra tmp1 - sonuç	-> bu y1
						mstep 2 : x1 * sin			-> tmp3'e yazılır
						mstep 3 : x2 * cos, sonra tmp3 + sonuç	-> bu y2

			ADIM 3 : ST_DRAIN
			Kısa Açıklama : Dönmüş vektörü qkv_proj ile aynı şekilde dışarı akıtıyor.

		Not: çift eşleme şeması (x1, x1 + D_MODEL/2 ile eşleniyor, "rotate half" tarzı) kendi varsayımım,
		     plan dokümanında golden modelin bunu mu yoksa komşu eleman eşleme tarzını mı (x1, x1+1 ile
		     eşlenir) kullandığı hiç yazmıyor. Pytorch golden model scripti ne yapıyorsa ona göre kontrol
		     edilmesi lazım, bu tür şeyler sessizce yanlış olup hızlı bir testte "doğru görünebiliyor".

		Not: sin_rom ve cos_rom $readmemh ile henüz var olmayan dosyalardan yükleniyor. Birinin standart
		     base 10000 rope frekans formülünü kullanarak her pozisyon ve her çift index'i için
		     sin(pos*freq_i) ve cos(pos*freq_i) değerlerini bf16 hex olarak döküp çıkaran küçük bir python
		     scripti yazması lazım.

		Not: henüz bir bf16_sub modülü yok, çıkarma işlemi burada ikinci operandın sign bitini flip'leyip
		     bf16_add'e vererek yapılıyor.

	--------------------------------------------------------------------------------------------------------------------------------------

	"gqa_mapper.sv"

	Fonksiyon Tanımı:

		gqa_mapper.sv küçük, tamamen combinational bir modül, hiç vektör verisine dokunmuyor. Sadece şu anda
		işlenen Q head'inin index'ini alıp o Q head'inin hangi K/V head'inden okuması gerektiğini
		döndürüyor. Örnek, 8 Q head ve 2 KV head varsa Q head 0'dan 3'e kadar hepsi KV head 0'a, Q head
		4'ten 7'ye kadar hepsi KV head 1'e eşlenir.

		PARAMETRELER:

			NUM_Q_HEADS		- modelin kaç tane query head'i var
			NUM_KV_HEADS		- modelin kaç tane key/value head'i var, NUM_Q_HEADS'i tam bölmeli

		PORTLAR:

			q_head_idx_i		- şu anda hangi Q head işleniyor
			kv_head_idx_o		- o Q head'in hangi KV head'den okuması gerektiği

		Not: bu da kendi varsayımım, plan dokümanı bu modülü "head broadcast/select muxing" diye
		     tanımlıyor, bu K/V veri bus'larını fiziksel olarak mux'laması gerektiği anlamına da gelebilir.
		     Ben sadece index dönen versiyonu seçtim çünkü Taha'nın kv_cache'i zaten adres için veri dışarı
		     şeklinde kurulu, yani "broadcast" her Q head için aynı adresi okumakla kendiliğinden oluyor.
		     Taha'nın kv_cache okuma portu netleşince teyit edilmesi lazım.

	--------------------------------------------------------------------------------------------------------------------------------------

	HER TAKIM ARKADAŞINDAN NEYE İHTİYACIM VAR

		BELİNAY'DAN:

			- bf16_add ve bf16_mul için donmuş bir port listesi, benimkiler zaten a_in, b_in,
			  result_sum_o ve result_mul_o varsayıyor, bu isimler ya da genişlikler değişirse benim
			  instantiate ettiğim yerler bozulur
			- bf16_add/bf16_mul'ün 0 cycle latency mi kalacağı yoksa pipeline mi edileceği konusunda net
			  bir cevap, bu qkv_proj.sv ve rope.sv'deki ST_COMPUTE'un kaç bekleme cycle'ı gerektirdiğini
			  değiştiriyor
			- qk_array.sv'nin port listesi, modüllerim onu doğrudan instantiate etmese de girdi formatını
			  bilmem lazım ki qkv_proj.sv'in (Q instance'ı) y_data_o akışı onun beklediğiyle uyuşsun

		TAHA'DAN:

			- qkv_proj.sv instance'larımın token vektörünü çekeceği dbuf.sv okuma arayüzü, şu an
			  x_data_i/x_valid_i/x_last_i üzerinde genel bir valid/ready stream tahmin ettim
			- kv_cache.sv okuma adres formatı, böylece gqa_mapper.sv'in kv_head_idx_o çıktısı Taha'nın
			  kuracağı adresleme şemasıyla (katman başına, head başına offset) gerçekten uyuşsun
			- top_fsm.sv'in faz sinyalleri, modüllerimin en üst seviyeden nasıl "şimdi yükle", "şimdi
			  hesapla", "şimdi akıt" denildiğini bilmem lazım, şu an dışarıdan bir start sinyali olmayan
			  kendi başlarına çalışan state machine'ler, 4'ten fazla instance doğru sırada çalışması
			  gerektiğinde bu yetmeyecek

		HASAN'DAN:

			- Hasan'dan doğrudan modüllerime bir şey akmıyor ama softmax.sv'in çıkış zamanlaması
			  (weights) score çarpı V adımının ne zaman başlayabileceğini etkiliyor, bu da Wo
			  qkv_proj.sv instance'ımın ne zaman başlayabileceğini etkiliyor, o yüzden softmax.sv iskelet
			  değil gerçek sayılarla çalışırken kabaca kaç cycle sürdüğünü bilmem lazım

	--------------------------------------------------------------------------------------------------------------------------------------

	BURADAN SONRA NASIL İLERLEYECEĞİM

		ADIM 1 : Yukarıdaki açık soruları Belinay ve Taha'ya gönder, boş beklemeden kendi tarafımda
			 varsayımları dokümante ederek çalışmaya devam et.

		ADIM 2 : rope_sin_rom.mem / rope_cos_rom.mem'i üretecek scripti yaz (python, numpy), böylece
			 rope.sv tanımsız bellek okumak yerine gerçekten simüle edilebilsin.

		ADIM 3 : Üç modül için de küçük standalone testbench yaz, hızlı bir python referansa karşı
			 (qkv_proj için düz numpy matmul, rope için numpy rotasyon formülü, gqa_mapper için
			 basit index matematiği), bu plan dokümanındaki Sprint 1 milestone'u, 23 Temmuz'a kadar
			 her şey cross module bağlantı olmadan önce kendi testbench'ini geçmiş olmalı.

		ADIM 4 : Belinay bf16 latency kontratını onaylayınca, mul/add artık 0 cycle değilse
			 qkv_proj.sv ve rope.sv'deki ST_COMPUTE zamanlamasını düzelt.

		ADIM 5 : Taha'nın dbuf ve kv_cache arayüzleri donunca, tahmin ettiğim port isimlerini
			 gerçekleriyle değiştir, gqa_mapper.sv'in çıktısını gerçek bir kv_cache okuma adresine
			 bağla.

		ADIM 6 : fpt-Can üzerinde her parça test edilebilir hale geldikçe küçük küçük commit at,
			 her şeyi biriktirip tek seferde atma, branch geçmişi Belinay/Taha/Hasan'ın neyin
			 neden değiştiğini görebileceği kadar okunaklı kalsın.

	--------------------------------------------------------------------------------------------------------------------------------------

	TAKIMLA TEYİT EDİLMESİ GEREKEN AÇIK VARSAYIMLAR

		1. rope.sv çift eşleme şeması, rotate-half mi komşu eleman eşleme mi		- golden model'e göre teyit et
		2. bf16_add/bf16_mul'ün 0 cycle latency kalması				- ST_COMPUTE zamanlamasına güvenmeden önce Belinay ile teyit et
		3. qkv_proj.sv weight_mem ve rope.sv sin_rom/cos_rom'un asenkron okunması	- BRAM inference kabul edilebilir mi yoksa register mı gerekiyor teyit et
		4. gqa_mapper.sv'in sadece index dönmesi, gerçek K/V veriyi mux'lamaması	- kv_cache okuma adreslemesinin nasıl çalıştığını Taha ile teyit et
		5. rope_sin_rom.mem / rope_cos_rom.mem üretim scripti				- henüz yok, rope.sv simüle edilmeden önce yazılması lazım

	--------------------------------------------------------------------------------------------------------------------------------------
