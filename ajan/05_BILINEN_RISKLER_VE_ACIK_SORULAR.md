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
- Drive mount, §2'nin `/content` ve Drive zip yolları, yükleme penceresi, download, `flush_and_unmount`;
- bf16, TF32, `adamw_torch_fused`, `dataloader_num_workers=2`;
- gerçek BGE-M3, bge-reranker-v2-m3 ve e5-small modelleri; SentencePiece belirteçleri ve e5 önek dalı;
- 512 belirteç uzunluğu (kuru çalıştırma 256 kullandı);
- hücre başına GPU tepe belleği (`CELL_LOG` kancaları, `reset_peak_memory_stats`; kuru çalıştırmada `gpu_tepe_GB` =
  `null`);
- tam veri (kuru çalıştırma 60 anchor, 24 val ve 27 test sorgusu kullandı).

| Risk | Belirti | Ne yapılır |
|---|---|---|
| **Sabit sürümler Colab imajıyla çakışabilir.** `faiss-cpu==1.15.1` ve diğerleri hiç denenmedi | §0'da pip hatası ya da sürüm uyarısı | Oturumu yeniden başlat. Sürümü güncellersen kuru çalıştırmayı tekrarla |
| **HF indirmesi** (≈5 GB, token kodu yok) | §5, §7 ya da §9'da istisna | Ağ veya geçici HF sorunu: hücreyi tekrar çalıştır |
| **`drop_amp_wrapper` accelerate'in iç adlarına bağlı** (`forward`, `_original_forward`). CPU'da bf16 ile doğrulandı | Hata yok. Parite <%100; öznitelikler bf16 hesaplanmış olur | Sarmalayıcı eğiticiye verilen nesneye takılır: `BI` ve `CE`'nin kendisine (`CE.model` iç HF modelidir, orada olmaz). `[type(m).__name__ for m in X.modules() if 'forward' in m.__dict__]` hem `X = BI` hem `X = CE` için boş liste vermeli |
| **CE LambdaLoss belleği.** ST 6.1 tüm parçaların logit'lerini birleştirip tek geri yayılım yapar; aktivasyon ∝ 8×16 çift × 512 | §9'da CUDA OOM | `CE_BATCH=4` ya da `CE_DOCS_PER_QUERY=8` (`02` §4). Önce belleği boşalt (tablonun altı) |
| **CoSENT önbelleksiz.** 128 dizi × 512 belirteç | §7.2'de OOM (L4/T4'te olası) | `BI_BATCH=32` ya da `BI_USE_COSENT=False`. Önce belleği boşalt (tablonun altı) |
| **fp16 dışa aktarım.** Bellekteki BI/CE de yuvarlanır; LTR yuvarlama öncesi özniteliklerle eğitildi; yeni bir süreç TF32 açmazsa tam fp32 hesaplar | Parite <%100 | Kabul edilebilir eşik koy (örn. Top-1 ≥ %95) ya da `EXPORT_HALF=False` (paket ≈4,6 GB) |
| **`import faiss` dışa aktarımın ortasında.** Başarısız olursa `config.json`, `run_summary.json`, zip ve Drive kopyası yazılmaz | §13'te ImportError | faiss çıkarımda kullanılmıyor; satır try içine alınabilir (`scripts/07`'de, sonra `08`) |
| **Drive kopyası try içinde değil.** Kota dolarsa OSError verir. `flush_and_unmount`'tan sonra Drive bağlı değildir | §13'te hata; sonrası FileNotFoundError | Kotayı kontrol et; Drive'ı yalnız yeniden bağla (`from google.colab import drive; drive.mount("/content/drive")`), sonra §13'ü yeniden çalıştır. §2'yi yeniden çalıştırma: `importlib.reload(core)` yapar ve §13'teki `joblib.dump(ATTR)` `PicklingError` verir |
| **Drive klasörü çalıştırmalar arasında temizlenmez.** `MyDrive/cv_ilan_sota/` her çalıştırmada aynı adlarla üzerine yazılır; `figures/` birleştirilir, eski PNG'ler kalır | Bir çalıştırmanın dosyaları ötekine karışır. §13'e ulaşmayan bir çalıştırmada klasördeki her şey öncekine aittir | Önceki çalıştırmanın dosyalarını yeni çalıştırmadan **önce** indir ya da kopyala. Aynı Colab makinesinde `!rm -rf /content/outputs` (yoksa eski görseller pakete ve Drive'a karışır). `runs/`'a koymadan `run_summary.json` → `paket_parmak_izi`, `cfg`, `toplam_sure_dk`'yı `.ipynb` çıktılarıyla eşleştir (`02` §7) |
| **Bayat paket.** §2, `CV_DATA_BUNDLE` yoksa `/content/data_bundle.zip`, çalışma klasörü ve `MyDrive/data_bundle.zip` arasından en yeni zip'i seçer, açılmış `/content/data_bundle` ile karşılaştırır (`metadata.json` + çekirdek baytları) ve farklıysa yeniden açar. Yükleme penceresi yalnız hiç zip yokken ve paket daha önce açılmamışken çıkar; aynı adı ikinci kez `data_bundle (1).zip` diye kaydeder, o okunmaz | Değişikliğin hiç etkisi yok gibi görünür; §2'deki `paket parmak izi` repodakinden farklı ya da `UYARI: zip bulunamadı; …` | Yeni zip'i §2'den önce Dosyalar paneliyle `/content/data_bundle.zip` olarak koy. Parmak izini `sha256sum data_bundle/metadata.json match_candidate_sota.py \| cut -c1-12` ile karşılaştır; `Çekirdek modül sürümü` sabit `1.0.0`'dır, bayatlığı göstermez. Çekirdek değiştiyse veri de değişmiş olabilir (`05` veriyi çekirdekle üretir): `02` §1 adım 1–3 |
| **§12.1 dikkat hücresi.** SentencePiece (`▁`) dalı kuru çalıştırmada denenmedi | Garip kelime birleştirme; çökme olası değil | Görsel çıktıyı kontrol et |
| **Optuna süresi ölçülmedi.** `LTR_TRIALS` × `LTR_FOLDS` = 40 × 5 = 200 LightGBM eğitimi CPU'da (erken durdurmalı, en çok `LTR_MAX_ROUNDS` = 2000 tur). Rehberdeki "3–5 dk" de tahmindir | Uzun bekleme | Kabul et ya da `LTR_TRIALS`'ı düşür (daha dar arama). Gerçek süre: ilk çalıştırmanın `hucre_sureleri` → §10'un `from sklearn.model_selection import GroupKFold` hücresi |
| **Resume yok** | Bağlantı koparsa her şey baştan | `02` §5 (altyapı ekleme kararı senin) |

**OOM'dan sonra** `free_mem()` tek başına yetmez: IPython'un sakladığı hata izi (`sys.last_*`, Python 3.12+'da
`sys.last_exc`, `InteractiveTB.tb`, `last_execution_result`) model, trainer ve aktivasyonları canlı tutar; §9'da `CE`
ve `ce_trainer` ayrıca globaldir. CFG'yi değiştir, sonra yeni bir hücrede `02` §4'teki bloğu çalıştır:
`del ce_trainer, CE` (yalnız §9; §7.1/§7.2'de model ve trainer yereldir, `BI`'yi silme),
`sys.last_traceback = sys.last_value = sys.last_type = None`, `if hasattr(sys, "last_exc"): sys.last_exc = None`,
`ip = get_ipython(); ip.InteractiveTB.tb = None; ip.last_execution_result = None`, `free_mem()`. Bellek inmezse
çalışma zamanını yeniden başlat; resume olmadığı için her şey baştan çalışır (`02` §4).

## B. Bilinen ve kabul edilmiş yöntem riskleri

Kaynak parantez içindedir. Kaynağı yazılmayan sayıların rapor dosyası yoktur; 2026-10-16'da paketten
(`data_bundle/*.parquet`, `annotation/qrels_validation_key.json`) yeniden üretildiler. B7 hariç: oradaki sayılar
örnekleyici simülasyonundandır ve yeniden üretilmedi.

1. **Etiketler gümüş standarttır.** LLM anotasyonuyla üretildiler; uzman insan yoktur.
   - IAA κ: ISCO 0,916, yaka 0,980, kıdem (ağırlıklı) 0,959. Uzmanlık adında birebir uyum %61,9 (grade 3 buna
     dayanır) (`reports/annotation_iaa.json`).
   - Kural derecelerinin kör doğrulaması (n = 360): kural grade-3 çiftlerinin 47/90'ı 2, kural grade-2 çiftlerinin
     43/90'ı 0–1 bulundu. İkinci dereceden κ 0,836 (`reports/qrels_validation.json`).
   - Sapmanın sistemleri eşit etkilediği **kanıtlanmadı**. Örneğin kural qrels'iyle eğitilen LTR, kurala benzeyen
     eşleşmeleri kayırabilir.
2. **Doğrulama örneklemi artık held-out değil.** Örneklem önceki bir bölmede çizildi. Mevcut bölmede çiftlerin 244'ü
   train, 69'u test, 47'si val tarafında. Yargıçlara rubrik verildi; doğrulama kuralın tutarlı uygulanıp
   uygulanmadığını sınar, kuralın doğruluğunu sınamaz. Grade-0 çiftler rastgele seçildi; zor negatiflerdeki hata
   oranı ölçülmedi.
3. **Zor negatiflerin bir kısmı yakın-alakalıdır.**
   - `neg_lexical` satırlarının 291/4.883'ü, `neg_bm25` satırlarının 279/4.883'ü anchor'la aynı ISCO-3 grubunu ve
     yakayı paylaşır, |kıdem| ≤ 1'dir.
   - Kıdem/yaka tuzaklarının 1.730/4.883'ü grade 1'dir ve MNRL negatifi olarak kullanılır (`metadata.json` →
     `triplet.negatif_grade_dagilimi`; notebook bunu yazdırmaz).
   - En çok kullanılan tuzak ilanı 114 satırda geçer (`metadata.json` → `triplet.train_istatistik`), çünkü
     `NEG_USE_CAP=25` bütün adaylar sınırı aşınca baypas edilir.
4. **Sentetik hedefler önceki bölmeden kalma.** Kayıtlı 868 hedef önceki bölmede çizildi; mevcut bölmede aynı koşul
   890 hedef verir (`scripts/06_synthetic_cvs.py` başlığı). Kabul edilen 710 sentetik CV'den
   (`reports/synthetic_cv_stats.json`) 54'ünün uzmanlığında artık ≥ 2 gerçek train CV'si var.
   - 158 sentetik CV, val/test'te de geçen bir uzmanlığı hedefler. Unvanlar kapı gereği ayrıktır. Kapı metin
     benzerliğini yalnız teste karşı ölçer: 710 sentetik CV'nin test CV'lerine en yüksek TF-IDF kosinüsünün p99'u
     0,169'dur (`reports/faz2_veri_ozeti.json` → `sizinti_kontrolleri.sentetik_test_max_cos_p99`). Val'e karşı
     benzerlik ve 158'lik alt küme ayrıca ölçülmez.
   - Vitrin 10587'nin train'deki tek kapsaması sentetik bir CV'dir.
5. **Val yeniden kullanılır.** Val etiketleri sırayla şunlar için kullanılır:
   - τ seçimi;
   - Aşama 1 erken durdurma ve en iyi checkpoint;
   - CE erken durdurma ve en iyi checkpoint;
   - LambdaMART eğitimi ve Optuna.

   §10.1 val'de tam hattı ölçer (`VAL_TBL`, `VAL_COMP`, `per_query_val.csv`; WARM havuz, LambdaMART kat dışı), ama
   temiz bir dev metriği değildir: aynı val yukarıdaki seçimlerde kullanıldı, LTR hiperparametreleri de aynı val
   CV'siyle seçildi. Sayılar hafifçe iyimserdir; değişiklikleri birbiriyle kıyaslamak içindir, yansız kestirim
   değildir. `ltr_cv_ndcg10` (WARM havuz) 40 denemenin en iyisi olduğu için ayrıca iyimserdir. Val sinyalleri WARM
   havuzda, test metrikleri tam havuzdadır; aynı ölçekte değildirler. Hiç kullanılmamış ayrı bir dev kümesi yoktur.
6. **CE eğitim listeleri ince ayarlı bi-encoder'ın kendi eğitim anchor'larından çıkar.** Bu anchor'lar Aşama 1'de
   ezberlendiği için train listeleri val/test listelerinden kolaydır; dağılım uyumsuzluğu olasıdır. Alternatifler:
   base model ya da BM25 adayları, veya K-katlı çapraz uydurulmuş bi-encoder.
7. **CoSENT batch'lerinde aynı CV'nin iki çifti hiç bulunmaz.** Simülasyonda 0 bulundu. `NoDuplicates` sentence1'e
   göre ayıklar; CoSENT sorgular arası kosinüs kalibrasyonu öğretir, sorgu içi derece sırası öğretmez. RoundRobin
   CoSENT çiftlerinin yalnız ≈%51'ini her epoch'ta gösterir.
8. **τ taraması gürültülüdür.** Her τ için en az 39 adım (`TEMP_FRACTION` = 0,5 × 4.883 MNRL satırı / `BI_BATCH` = 64),
   1 epoch, CoSENT yok.
9. **Belirteç kırpılması.**
   - `cv_struct`'ın %23,4'ü ve `job_struct`'ın %28,2'si CE için 1.400 karakterde kesilir; çift ayrıca 512 belirteçte
     kesilir.
   - XLM-R belirteç uzunlukları burada ölçülemedi (HF erişimi yok); §9.1 ilk çalıştırmada ölçer (`tani` →
     `kirpilma_*`).
10. **Lokasyon grade 3'ü belirler.** Aynı uzmanlık/yaka/kıdem çiftlerinin val'de %71,3'ü, testte %69,9'u yalnız şehir
    yüzünden grade 2'dir (`reports/metric_ceilings.json` → `grade3_lokasyon_bagimliligi`). Bu bir alaka tanımı
    tercihidir; değiştirmek kullanıcı kararıdır.
11. **Teknoloji yığını ayrımı yok.** "Java backend" ile ".NET backend" aynı uzmanlıktır. Rehber BÖLÜM 5'te yazılı.
12. **Bazı CV metinlerinde sektör enum kodu geçiyor.** 15 gerçek CV'nin `cv_struct` metninde
    "Deneyim: <unvan>: <KOD> sektöründe …" şablonu var (train 9, val 5, test 1). Yalnız 3'ünde kod CV'nin kendi
    `gold_industry`'sidir (train 1, val 1, test 1); 12'sinde başka bir sektör yazar (ör. "Matbaa İşçisi:
    CREATIVE_DESIGN sektöründe", altın etiket INDUSTRIAL_MANUFACTURING). Yani bu sızıntıdan çok etiket gürültüsüdür;
    sektör sınıflandırıcısı (`AttributeModels`, `cv_struct` üzerinde) onu az sayıda örnekte kullanabilir. Ayrıca 30
    CV'de "HR" geçer ("HR Business Partner", "HR Metrics", "Logo Tiger HR" gibi; 27'si HR sektörü CV'si). Bu doğal
    bir kısaltmadır, kod sızıntısı değildir; metinden silinmemelidir. Etki küçüktür.
13. **kNN etiket aktarımı (aday kaldıraç) yeni ilanlara karşı yanlıdır.** Cold ilanlara hiç sinyal vermez. Ürün
    kararıdır, bkz. `00` §4.

## C. Açık sorular

Önce ölçüm, sonra karar. Karar sinyalleri yalnız val'dir ve WARM havuzda ölçülür (`03` §A); test tabloları yalnız
kayıttır.
- [ ] Gerçek A100 süresi ve tepe bellek bölüm bölüm nedir? `run_summary.json` → `hucre_sureleri`: CFG hücresinden
      sonraki her hücrenin ilk satırı, `sn` ve `gpu_tepe_GB` (tepe her hücrenin başında sıfırlanır). §0 kurulum ve
      içe aktarma, CFG hücresi, §13 dışa aktarım ve sonrası kayda girmez. `gpu_tepe_bellek_GB` hücre tepelerinin en
      büyüğüdür; `sureler_sn` yalnız eğitim koşularını tutar (`τ=…`, `Bi-encoder`, `Cross-encoder`); toplam:
      `toplam_sure_dk`.
- [ ] Hibrit aday kümesinin val tavanı (§9.1 → `tani` → `aday_tavani_val(WARM)`; cold ilanlar ne adayda ne
      qrels'te) nerede? Eşik (`04` §1 düğüm 3): < 92 → Aşama 2 işine başlama, önce Aşama 1 / `N_CANDIDATES` / RRF;
      92–95 → gri bölge, Aşama 1 / aday iyileştirmesi muhtemelen hâlâ en verimlisi; ≥ 95 → Aşama 2 işi karşılık
      verebilir. Kıyas: val WARM BM25 ilk-50 84,14 / ilk-100 90,67 (`reports/metric_ceilings.json` →
      `val|WARM|bm25_ilk{K}_kahin_yeniden_siralama`). Val tavanları test tavanlarından birkaç puan düşüktür;
      `aday_tavani_test(tam_havuz,yalniz_rapor)` yalnız rapor içindir. Tavan düşükse `N_CANDIDATES`, RRF ağırlığı ve
      dense geri getirmeden hangisi en ucuz kazancı verir? RRF ağırlığı değişirse `EXPORT_CFG`'ye ve
      `SOTAMatcher.match`'e de girer (çekirdek değişikliği); yoksa paket notebook'tan farklı sıralar.
- [ ] BI ve CE için 512 belirteç kırpılma oranı nedir? (§9.1 → `tani` → `kirpilma_bi_ilan`, `kirpilma_bi_cv`,
      `kirpilma_ce_cift`; örneklem: 400 ilan, ilk 400 val CV'si, 40 val sorgusu × ilk 10 aday.)
- [ ] Gerçek modellerle parite ne çıkıyor? Bir eşik konmalı mı?
- [ ] Sabit sürümler güncel Colab imajına yeniden başlatma gerektirmeden kuruluyor mu?
- [ ] Leksikal kaldıraçların (ISCO-4 özniteliği, kNN, K=100) val kazancı dense ve CE öznitelikleri eklenince de
      sürüyor mu? (`reports/lexical_lever_probe_val.json` yalnız leksikal hatta, val WARM havuzda ölçüldü. Yeniden
      ölçmek için taze klonda önce `unzip -o data_bundle.zip`.)
- [ ] τ taraması bir şey söylüyor mu, yoksa τ sabitlenmeli mi? (`val_sinyalleri` → `temperature_sweep`)
- [ ] CoSENT ablation'ı (`BI_USE_COSENT=False`) val'de ne veriyor?
- [ ] Sentetik veri ablation'ı (`USE_SYNTHETIC=False`) val'de ne veriyor?

İki ablation da iki çalıştırmanın §10.1 `VAL_TBL`'si ve `per_query_val.csv` üzerinde eşleştirilmiş randomizasyonla
kıyaslanır; §11 test tabloları yalnız kayıttır.

Kullanıcı kararı bekleyenler:
- [ ] P@10 hedefi yeniden ifade edilsin mi? Seçenekler: P@10_capped > 85, ya da "kâhine göre oran" (örn.
      P@10 / 83,38; testteki kâhin P@10'u, etiketlerin yapısı, `reports/metric_ceilings.json`).
- [ ] kNN etiket aktarımı ürün için kabul edilebilir mi?
- [ ] CV metinleri dış bir LLM'e gönderilsin mi (yeniden sıralama; gizlilik)? Yeni sentetik veri ya da LLM ile
      yeniden anotasyon yapılsın mı (maliyet)? Üretim yalnız ilan tarafındaki WARM bilgiyi kullanır; hedefler teste
      bakılarak seçilmez.
- [ ] GPU bütçesi nedir (A100 saati / tam çalıştırma sayısı)? Cevap `06` → "Kullanıcı kararları kaydı"na yazılır.
      Tam çalıştırma süresi ölçülmedi (notebook başlığındaki ≈1,5–2,5 saat tahmindir); ilk çalıştırmanın
      `toplam_sure_dk`'sı bütçeyi hesaplamaya yarar.

Bu listede olmayan: ilan tarafı `gold_*` etiketlerini öznitelik yapmak. Bu **asla** yapılmaz (değişmez kural 2,
`00` §4) ve kullanıcıya seçenek olarak sunulmaz. Kullanıcı ısrar ederse ayrı, açıkça etiketlenmiş bir sistem olur,
ana sonuç olmaz.
