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

	GÜNCELLEME - 07.08.2026

		Belinay FPT_belinay_codes'ta (commit e295872) bf16_mul ve bf16_add'i baştan yazdı: ikisi de artık 2 stage
		pipeline (valid_i/valid_o handshake ile, 0 cycle değil), ve portları 16-bit değil 17-bit oldu, bit[16]
		aritmetiğe hiç girmeyen bir "source tag" (kaynak etiketi) olarak pipeline boyunca değişmeden taşınıp
		çıkışta geri ekleniyor. Belinay'la konuşup 0 = benim (Can) bloğumdan geldi, 1 = onun bloğundan geldi
		diye karar verdik, ihtiyaç Belinay'ın qk_array'inin paylaşılan fiziksel kaynaklarını (aynı array hem
		QK^T hem Score×V için tekrar kullanılıyor) hangi işlemin beslediğini karışıklık olmadan ayırt edebilmesi.

		Bunun üzerine qkv_proj.sv ve rope.sv'yi güncelledim:
			- y_data_o artık [DATA_WIDTH:0] (17 bit), bit DATA_WIDTH sabit SRC_TAG_CAN (0), geri kalanı
			  eskisi gibi bf16 değeri. x_data_i 16 bit kaldı, gelen verinin tag'i benim için önemli değil,
			  bu modül ne üretirse onu kendi tag'imle işaretliyorum.
			- ST_COMPUTE (qkv_proj) / mstep'in çarpma-toplama kısmı (rope) artık tek cycle'da bitmiyor,
			  bf16_mul/bf16_add'in valid_i/valid_o handshake'ini bekleyen ISSUE/WAIT alt state'lerine
			  bölündü, detaylar aşağıdaki STATE MACHINE bölümlerinde güncellendi.
			- gqa_mapper.sv'ye dokunmadım, bf16 birimi kullanmıyor, tag kavramı ona hiç uygulanmıyor.

		Açık nokta: qk_pe.sv içinde `assert(q_in[16]==k_in[16])` var, yani bir PE'ye giren iki operand aynı
		tag'i taşımak zorunda. Q×K^T geçişinde ikisi de benim rope.sv'imden geldiği için otomatik uyuşuyor.
		Score×V geçişinde (array'in ikinci kullanımı) bir operand Hasan'ın softmax çıkışı, diğeri benim V
		çıkışım olacak, o ikisinin qk_array'e girerken aynı tag'i taşıyacak şekilde kimin bağlayacağı henüz
		netleşmedi (muhtemelen top_fsm.sv'i yazan kişinin işi). Ayrıca Belinay'ın yeni qk_array (2).sv'si artık
		paralel bus değil, derinlik dilimi başına SIZE seri Q/K çifti bekliyor - benim rope.sv/qkv_proj.sv'imin
		tek-vektör-seri çıkışını o formata paketleyecek bir ara katman (buffer/FIFO) henüz yazılmadı.

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
			x_data_i, x_valid_i, x_last_i	- girdi vektörü, clock başına bir eleman akıyor, 16 bit (tag yok,
							  gelen verinin nereden geldiği bu modül için önemli değil)
			y_data_o, y_valid_o, y_last_o	- çıkış vektörü, clock başına bir eleman akıyor, y_data_o artık
							  17 bit: bit[DATA_WIDTH] sabit SRC_TAG_CAN (0), bit[DATA_WIDTH-1:0]
							  bf16 değeri
			busy_o				- modül idle değilken yüksek, sadece waveform debug için

		STATE MACHINE ADIMLARI: (07.08.2026 güncellendi, Belinay'ın pipeline'lı bf16_mul/bf16_add'ine göre)

			ADIM 1 : ST_LOAD
			Kısa Açıklama : x_ready_o yüksekken girdi vektörünü clock başına bir eleman olarak x_mem'e
					 alıyor. x_last_i geldiğinde modül ST_MUL_ISSUE'ya geçiyor.

			ADIM 2 : ST_MUL_ISSUE / ST_MUL_WAIT
			Kısa Açıklama : mac_idx_q'daki weight*x çarpımı için bf16_mul'a valid_i pulse'lanıyor
					 (ST_MUL_ISSUE), sonra valid_o gelene kadar bekleniyor (ST_MUL_WAIT), sonuç
					 prod_q'ya kaydediliyor.

			ADIM 3 : ST_ADD_ISSUE / ST_ADD_WAIT
			Kısa Açıklama : acc_q + prod_q toplamı için bf16_add'e valid_i pulse'lanıyor, valid_o
					 gelince ya bir sonraki mac_idx'e geçiliyor (acc_q güncellenip ST_MUL_ISSUE'ya
					 dönülüyor) ya da mac_idx_q son elemansa sonuç y_mem'e yazılıp out_idx_q
					 ilerletiliyor. Her çıkış elemanı artık D_MODEL clock değil, D_MODEL * (mul+add
					 pipeline gecikmesi) kadar cycle sürüyor.

			ADIM 4 : ST_DRAIN
			Kısa Açıklama : y_ready_i yüksekken bitmiş y_mem içeriğini clock başına bir eleman dışarı
					 akıtıyor, y_last_o son elemanı işaretliyor. Değişmedi.

		Not: weight_mem düz bir array index ile okunuyor, bu asenkron bir okuma, Taha'ya dbuf.sv için
		     söylediğim aynı endişe burada da geçerli. İlk versiyon için böyle bırakıldı, önce doğruluk.

		Not: bf16_mul ve bf16_add artık 0 cycle latency değil (Belinay'ın e295872 güncellemesi), bu yüzden
		     yukarıdaki ISSUE/WAIT alt state'leri eklendi. Eski "0 cycle varsayımı" notu artık geçerli değil.

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
			x_data_i, x_valid_i, x_last_i	- girdi vektörü, clock başına bir eleman akıyor, 16 bit (tag yok,
							  qkv_proj.sv'deki gerekçenin aynısı)
			y_data_o, y_valid_o, y_last_o	- dönmüş çıkış vektörü, clock başına bir eleman akıyor, y_data_o
							  artık 17 bit: bit[DATA_WIDTH] sabit SRC_TAG_CAN (0). Q ve K aynı
							  modülden geçtiği için ikisi de aynı tag'i taşıyor, Belinay'ın
							  qk_pe.sv'i tam bunu bekliyor (q_in[16]==k_in[16] assert'i var)

		STATE MACHINE ADIMLARI: (07.08.2026 güncellendi, Belinay'ın pipeline'lı bf16_mul/bf16_add'ine göre)

			ADIM 1 : ST_LOAD
			Kısa Açıklama : qkv_proj ile aynı mantık, önce bütün girdi vektörünü x_mem'e topluyor, ayrıca
					 vektörün ilk elemanında pos_i'yi pos_q'ya latch'liyor.

			ADIM 2 : ST_MUL_ISSUE / ST_MUL_WAIT / ST_ADD_ISSUE / ST_ADD_WAIT
			Kısa Açıklama : Bir seferde bir çift üzerinde çalışıyor (pair_idx_q), her çiftin içinde de dört
					 mikro adım (mstep_q 0'dan 3'e) çalıştırıyor, çünkü rotasyon dört çarpma iki
					 toplama gerektiriyor ama elimizde sadece tek bir paylaşılan çarpıcı ve tek bir
					 paylaşılan toplayıcı var. Her mikro adım artık bf16_mul'a valid_i pulse'layıp
					 valid_o'yu bekliyor (ST_MUL_ISSUE/ST_MUL_WAIT); mstep 1 ve 3 çarpımdan sonra
					 ayrıca bf16_add'e valid_i pulse'layıp valid_o'yu bekliyor (ST_ADD_ISSUE/
					 ST_ADD_WAIT), mstep 0 ve 2 sadece çarpıp tmp'ye yazdığı için toplama adımına
					 hiç girmiyor.

						mstep 0 : x1 * cos				-> tmp1'e yazılır, toplama yok
						mstep 1 : x2 * sin, sonra tmp1 - sonuç		-> bu y1
						mstep 2 : x1 * sin				-> tmp3'e yazılır, toplama yok
						mstep 3 : x2 * cos, sonra tmp3 + sonuç		-> bu y2

			ADIM 3 : ST_DRAIN
			Kısa Açıklama : Dönmüş vektörü qkv_proj ile aynı şekilde dışarı akıtıyor. Değişmedi.

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

	"projection_block.sv" [YENİ - 07.08.2026, Taha'ya teslim edilecek]

	Fonksiyon Tanımı:

		Üç modülümü (qkv_proj x4, rope x2, gqa_mapper x1) tek kutuda toplayan üst seviye wrapper. Taha ile
		WhatsApp'ta netleşen karar şuydu: Taha'ya "büyük kutu, 5 sabit çıkış, routing yok" vermem gerekiyormuş,
		o "hangi çıkış nereye" kararını kendisi top seviyede mux/tag ile çözecek sanıyordu, ben de "wiring ile
		ayrılıyor, tag'e gerek yok" diyene kadar bu netleşmemişti. İçi Taha'yı ilgilendirmiyor, o yüzden içeriye
		hiç bakmadan doğrudan top.sv'ye takabileceği 5 çıkış var:

			q_data_o		-> Belinay'a (qk_array girişi)
			k_data_o		-> Taha'ya (kv_cache yazma)
			v_data_o		-> Taha'ya (kv_cache yazma)
			kv_head_idx_o		-> Taha'ya (kv_cache okuma adresi)
			out_data_o		-> Taha'ya (axi_stream_if, dışarı çıkış)

		Girişte de iki ayrı, hiç karışmayan port var:

			token_data_i		- dbuf'tan geliyor, Wq/Wk/Wv ile çarpılıyor (aynı token, 3 farklı weight)
			attn_data_i		- Belinay'dan (Score×V sonrası) geliyor, Wo ile çarpılıyor

		Hiçbir yerde "bu veri nereden geldi, nereye gidecek" diye karar veren bir mux yoktur, her şey hardwired.
		write enable de ayrı bir sinyal değil, k_valid_o/v_valid_o zaten valid/ready handshake'in kendisi,
		Taha valid yüksekken cache'e yazacak, last geldiğinde vektör bitmiş demek.

		Not [07.08.2026]: ilk yazdığım halinde token_ready_o sadece Wq instance'ının x_ready_o'suna
		bağlıydı, Wk ve Wv'ninki hiç dışarı çıkmıyordu. Üçü de aynı token_valid_i/token_data_i'yi paylaştığı
		için normalde lockstep çalışırlar, ama biri (mesela Wv) k_ready_i/v_ready_i backpressure'i yüzünden
		ST_DRAIN'de takılıp ST_LOAD'a geç dönerse, dışarısı token_ready_o'yu (sadece Wq'dan) hâlâ yüksek
		görüp yeni token gönderebilirdi, Wv de kendi state'i ST_LOAD olmadığı için o veriyi sessizce
		kaçırırdı. Düzeltildi: token_ready_o artık üçünün x_ready_o'sunun AND'i.

	--------------------------------------------------------------------------------------------------------------------------------------

	HER TAKIM ARKADAŞINDAN NEYE İHTİYACIM VAR

		BELİNAY'DAN:

			[07.08.2026 ÇÖZÜLDÜ] bf16_add/bf16_mul port listesi ve 0 cycle/pipeline sorusu: e295872'de
			netleşti, ikisi de artık clk_i/rst_ni/valid_i/valid_o eklenmiş 17-bit (a_in/b_in/result_*_o)
			2-stage pipeline modüller, bit[16] source tag. qkv_proj.sv ve rope.sv buna göre güncellendi.

			- Score×V geçişinde (qk_array aynı fiziksel array'i ikinci kez kullandığında) benim V
			  çıkışım ile Hasan'ın softmax çıkışının qk_array'e aynı tag ile girmesi lazım (qk_pe.sv'nin
			  q_in[16]==k_in[16] assert'i yüzünden). Bunu kim, nerede (top_fsm.sv'de mi?) garantileyecek,
			  netleşmedi.
			- qk_array (2).sv artık paralel bus değil, "derinlik dilimi" başına SIZE tane seri Q/K çifti
			  bekliyor (enable_i/start_i/valid_i/last_i handshake'i ile). Benim rope.sv/qkv_proj.sv'imin
			  tek-vektör-seri (D_MODEL eleman, bir seferde bir vektör) çıkışını bu SIZE-satır-genişliğinde
			  derinlik-dilimi formatına kim paketleyecek netleşmedi, muhtemelen ortada bir buffer/FIFO
			  katmanı gerekiyor ve kimin yazacağı konuşulmadı.

		TAHA'DAN:

			[07.08.2026 ÇÖZÜLDÜ] Q/K/V/Wo çıkışının nereye gideceğinin nasıl ayırt edileceği: WhatsApp'ta
			netleşti, ayırt etme yok, projection_block.sv 5 sabit çıkış veriyor (yukarıdaki bölüme bak),
			Taha kendi tarafında hardwired bağlayacak.

			- Taha'nın top.sv'si (FPT_taha, commit 8d76d14) şu an `mod_b` diye TEK bir instance + o tek
			  çıkışı tag ve eleman sayacıyla Q/V/K/FIFO'ya dağıtan mod_b_output_router.sv üzerine kurulu,
			  bu artık projection_block.sv'nin 5-port hardwired modeliyle uyuşmuyor. Taha'nın top.sv'yi
			  mod_b + mod_b_output_router yerine projection_block'u doğrudan 5 portla bağlayacak şekilde
			  güncellemesi lazım, bu benim tarafımda yapılacak bir şey değil.
			- kv_cache.sv okuma adres formatı, böylece gqa_mapper.sv'in kv_head_idx_o çıktısı Taha'nın
			  kuracağı adresleme şemasıyla (katman başına, head başına offset) gerçekten uyuşsun
			- projection_block.sv'nin ne zaman "şimdi başla" denileceği: şu an dışarıdan bir start sinyali
			  yok, sadece token_valid_i/attn_valid_i handshake'ine göre kendiliğinden çalışıyor, top.sv
			  seviyesinde ekstra bir faz kontrolüne ihtiyaç olup olmadığını Taha ile teyit etmek lazım

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
		2. [07.08.2026 ÇÖZÜLDÜ] bf16_add/bf16_mul'ün 0 cycle latency kalması		- Belinay pipeline'lı yaptı (e295872), qkv_proj.sv/rope.sv güncellendi
		3. qkv_proj.sv weight_mem ve rope.sv sin_rom/cos_rom'un asenkron okunması	- BRAM inference kabul edilebilir mi yoksa register mı gerekiyor teyit et
		4. gqa_mapper.sv'in sadece index dönmesi, gerçek K/V veriyi mux'lamaması	- kv_cache okuma adreslemesinin nasıl çalıştığını Taha ile teyit et
		5. rope_sin_rom.mem / rope_cos_rom.mem üretim scripti				- henüz yok, rope.sv simüle edilmeden önce yazılması lazım
		6. Score×V geçişinde qk_array'e giren V ile softmax çıkışının aynı source	- top_fsm.sv'i kim yazıyorsa onunla teyit et
		   tag'i taşıyıp taşımadığı (qk_pe.sv assert'i bunu istiyor)
		7. rope.sv/qkv_proj.sv'in tek-vektör-seri çıkışını qk_array (2).sv'nin		- Belinay ile kimin yazacağını netleştir
		   istediği "derinlik dilimi başına SIZE seri çift" formatına paketleyecek
		   ara katman (buffer/FIFO) henüz yok
		8. [07.08.2026 ÇÖZÜLDÜ] Q/K/V/Wo çıkışının ayırt edilmesi			- projection_block.sv 5 sabit hardwired çıkış veriyor, mux/tag yok
		9. Taha'nın top.sv'sinin mod_b + mod_b_output_router modelinin		- Taha'nın kendisiyle teyit et, benim tarafımda yapılacak iş yok
		   projection_block.sv'nin 5-port hardwired modeline göre yeniden
		   kablanması gerekiyor, henüz yapılmadı

	--------------------------------------------------------------------------------------------------------------------------------------
