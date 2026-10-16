# Bilinen riskler ve açık sorular

Üç bölüm var:
- **A:** GPU'da hiç çalışmamış kod yolları. İlk çalıştırmada bunlar patlayabilir.
- **B:** Bilinen ve kabul edilmiş yöntem riskleri. Savunmada söylenmesi gerekenler.
- **C:** Açık sorular. Cevabı ölçümle verilecek.

Önceki incelemelerde reddedilen bulgular da B'de "kabul edilmiş risk" olarak yer alıyor. Reddedilmeleri yok
sayıldıkları anlamına gelmez; kusur değil, bilinçli tercih olarak değerlendirildiler.

## A. GPU'da doğrulanmamış kod yolları

CPU kuru çalıştırması şunları **çalıştırmadı**:
- pip kurulumu;
- Drive mount, upload, download, `flush_and_unmount`;
- bf16, TF32, `adamw_torch_fused`, `dataloader_num_workers=2`;
- gerçek BGE-M3, bge-reranker-v2-m3 ve e5-small modelleri; SentencePiece belirteçleri ve e5 önek dalı;
- 512 belirteç uzunluğu (kuru çalıştırma 256 kullandı);
- tam veri (kuru çalıştırma 60 anchor, 24 val ve 27 test sorgusu kullandı).

| Risk | Belirti | Ne yapılır |
|---|---|---|
| **Sabit sürümler Colab imajıyla çakışabilir.** `faiss-cpu==1.15.1` ve diğerleri hiç denenmedi | §0'da pip hatası ya da sürüm uyarısı | Oturumu yeniden başlat. Sürümü güncellersen kuru çalıştırmayı tekrarla |
| **HF indirmesi** (≈5 GB, token kodu yok) | §5, §7 ya da §9'da istisna | Ağ veya geçici HF sorunu: hücreyi tekrar çalıştır |
| **`drop_amp_wrapper` accelerate'in iç adlarına bağlı** (`forward`, `_original_forward`). CPU'da bf16 ile doğrulandı | Hata yok. Parite <%100; öznitelikler bf16 hesaplanmış olur | `BI.forward` ve `CE.model.forward` örneğe bağlı mı kontrol et (`'forward' in m.__dict__`) |
| **CE LambdaLoss belleği.** ST 6.1 tüm parçaların logit'lerini birleştirip tek geri yayılım yapar; aktivasyon ∝ 8×16 çift × 512 | §9'da CUDA OOM | `CE_BATCH=4` ya da `CE_DOCS_PER_QUERY=8` (`02` §4) |
| **CoSENT önbelleksiz.** 128 dizi × 512 belirteç | §7.2'de OOM (L4/T4'te olası) | `BI_BATCH=32` ya da `BI_USE_COSENT=False` |
| **fp16 dışa aktarım.** Bellekteki BI/CE de yuvarlanır; LTR yuvarlama öncesi özniteliklerle eğitildi; yeni bir süreç TF32 açmazsa tam fp32 hesaplar | Parite <%100 | Kabul edilebilir eşik koy (örn. Top-1 ≥ %95) ya da `EXPORT_HALF=False` (paket ≈4,6 GB) |
| **`import faiss` dışa aktarımın ortasında.** Başarısız olursa `config.json` ve zip yazılmaz | §13'te ImportError | faiss çıkarımda kullanılmıyor; satır try içine alınabilir |
| **Drive kopyası try içinde değil.** Kota dolarsa OSError verir. `flush_and_unmount`'tan sonra Drive bağlı değildir | §13'te hata; sonrası FileNotFoundError | Kotayı kontrol et; tekrar için §2'yi yeniden çalıştır |
| **Çekirdek pakette, bayat paket sessizce kullanılır** | Değişikliğin hiç etkisi yok gibi görünür | `02` §1. `/content/data_bundle`'ı sil; `Çekirdek modül sürümü` satırına bak |
| **§12.1 dikkat hücresi.** SentencePiece (`▁`) dalı kuru çalıştırmada denenmedi | Garip kelime birleştirme; çökme olası değil | Görsel çıktıyı kontrol et |
| **Optuna süresi.** Tahmin 3–5 dk, yerel ölçüm ≈17 dk | Uzun bekleme | `LTR_TRIALS`'ı düşür ya da kabul et |
| **Resume yok** | Bağlantı koparsa her şey baştan | `02` §5 (altyapı ekleme kararı senin) |

## B. Bilinen ve kabul edilmiş yöntem riskleri

1. **Etiketler gümüş standarttır.** LLM anotasyonuyla üretildiler; uzman insan yoktur.
   - IAA κ: ISCO 0,916, yaka 0,980, kıdem (ağırlıklı) 0,959. Uzmanlık adında birebir uyum %61,9 (grade 3 buna dayanır).
   - Kural derecelerinin kör doğrulaması (n = 360): kural grade-3 çiftlerinin 47/90'ı 2, kural grade-2 çiftlerinin
     43/90'ı 0–1 bulundu. İkinci dereceden κ 0,836.
   - Sapmanın sistemleri eşit etkilediği **kanıtlanmadı**. Örneğin kural qrels'iyle eğitilen LTR, kurala benzeyen
     eşleşmeleri kayırabilir.
2. **Doğrulama örneklemi artık held-out değil.** Örneklem önceki bir bölmede çizildi. Mevcut bölmede çiftlerin 244'ü
   train, 69'u test, 47'si val tarafında. Yargıçlara rubrik verildi; doğrulama kuralın tutarlı uygulanıp
   uygulanmadığını sınar, kuralın doğruluğunu sınamaz. Grade-0 çiftler rastgele seçildi; zor negatiflerdeki hata
   oranı ölçülmedi.
3. **Zor negatiflerin bir kısmı yakın-alakalıdır.**
   - `neg_lexical` satırlarının 291/4.883'ü, `neg_bm25` satırlarının 279/4.883'ü anchor'la aynı ISCO-3 grubunu ve
     yakayı paylaşır, |kıdem| ≤ 1'dir.
   - Kıdem/yaka tuzaklarının 1.730/4.883'ü grade 1'dir ve MNRL negatifi olarak kullanılır.
   - En çok kullanılan tuzak ilanı 114 satırda geçer, çünkü `NEG_USE_CAP=25` bütün adaylar sınırı aşınca baypas edilir.
4. **Sentetik hedefler önceki bölmeden kalma.** Kayıtlı 868 hedef önceki bölmede çizildi; mevcut bölmede aynı koşul
   890 hedef verir. Kabul edilen 710 sentetik CV'den 54'ünün uzmanlığında artık ≥ 2 gerçek train CV'si var.
   - 158 sentetik CV, val/test'te de geçen bir uzmanlığı hedefler. Unvanlar kapı gereği ayrıktır; metin benzerliği
     düşüktür (p99 0,169).
   - Vitrin 10587'nin train'deki tek kapsaması sentetik bir CV'dir.
5. **Val yeniden kullanılır.** Val etiketleri sırayla şunlar için kullanılır:
   - τ seçimi;
   - Aşama 1 erken durdurma ve en iyi checkpoint;
   - CE erken durdurma ve en iyi checkpoint;
   - LambdaMART eğitimi ve Optuna.

   Notebook'ta LambdaMART'tan sonra temiz bir tam-hat dev metriği yoktur. Tek kat dışı sinyal `ltr_cv_ndcg10`'dur
   (WARM havuz; test ile aynı ölçekte değildir).
6. **CE eğitim listeleri ince ayarlı bi-encoder'ın kendi eğitim anchor'larından çıkar.** Bu anchor'lar Aşama 1'de
   ezberlendiği için train listeleri val/test listelerinden kolaydır; dağılım uyumsuzluğu olasıdır. Alternatifler:
   base model ya da BM25 adayları, veya K-katlı çapraz uydurulmuş bi-encoder.
7. **CoSENT batch'lerinde aynı CV'nin iki çifti hiç bulunmaz.** Simülasyonda 0 bulundu. `NoDuplicates` sentence1'e
   göre ayıklar; CoSENT sorgular arası kosinüs kalibrasyonu öğretir, sorgu içi derece sırası öğretmez. RoundRobin
   CoSENT çiftlerinin yalnız ≈%51'ini her epoch'ta gösterir.
8. **τ taraması gürültülüdür.** Her τ için ≈38 adım, 1 epoch, CoSENT yok.
9. **Belirteç kırpılması.**
   - `cv_struct`'ın %23,4'ü ve `job_struct`'ın %28,2'si CE için 1.400 karakterde kesilir; çift ayrıca 512 belirteçte
     kesilir.
   - XLM-R belirteç uzunlukları burada ölçülemedi (HF erişimi yok); §9.1 ilk çalıştırmada ölçer.
10. **Lokasyon grade 3'ü belirler.** Aynı uzmanlık/yaka/kıdem çiftlerinin %69,9'u yalnız şehir yüzünden grade 2'dir.
    Bu bir alaka tanımı tercihidir; değiştirmek kullanıcı kararıdır.
11. **Teknoloji yığını ayrımı yok.** "Java backend" ile ".NET backend" aynı uzmanlıktır. Rehber BÖLÜM 5'te yazılı.
12. **Bazı CV metinlerinde sektör kodu geçiyor.** 45 CV'nin metninde bir sektör enum kodu, 30'unda kendi
    `gold_industry` kodu geçer (train 20, val 3, test 7). Sektör sınıflandırıcısı bunu kullanabilir. Etki küçüktür
    ama vardır.
13. **kNN etiket aktarımı (aday kaldıraç) yeni ilanlara karşı yanlıdır.** Cold ilanlara hiç sinyal vermez. Ürün
    kararıdır, bkz. `00` §4.

## C. Açık sorular

Önce ölçüm, sonra karar:
- [ ] Gerçek A100 süresi ve tepe bellek bölüm bölüm nedir? (`run_summary.json`)
- [ ] Hibrit aday kümesinin val tavanı (§9.1) 92'yi geçiyor mu? Geçmiyorsa `N_CANDIDATES`, RRF ağırlığı ve dense
      geri getirmeden hangisi en ucuz kazancı verir?
- [ ] BI ve CE için 512 belirteç kırpılma oranı nedir?
- [ ] Gerçek modellerle parite ne çıkıyor? Bir eşik konmalı mı?
- [ ] Sabit sürümler güncel Colab imajına yeniden başlatma gerektirmeden kuruluyor mu?
- [ ] Leksikal kaldıraçların (ISCO-4 özniteliği, kNN, K=100) val kazancı dense ve CE öznitelikleri eklenince de
      sürüyor mu? (`reports/lexical_lever_probe_val.json` yalnız leksikal hatta ölçüldü.)
- [ ] τ taraması bir şey söylüyor mu, yoksa τ sabitlenmeli mi?
- [ ] CoSENT ablation'ı (`BI_USE_COSENT=False`) val'de ne veriyor?
- [ ] Sentetik veri ablation'ı (`USE_SYNTHETIC=False`) val'de ne veriyor?

Kullanıcı kararı bekleyenler:
- [ ] P@10 hedefi yeniden ifade edilsin mi? Seçenekler: P@10_capped > 85, ya da "kâhine göre oran" (örn.
      P@10 / 83,38).
- [ ] kNN etiket aktarımı ürün için kabul edilebilir mi?
- [ ] İlan tarafı altın ISCO/uzmanlık etiketleri hiç kullanılabilir mi? Şu an yasak.
- [ ] Dış LLM ile yeniden sıralama veya yeniden anotasyon (gizlilik + maliyet) yapılsın mı?
