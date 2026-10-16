# Bu proje için eğitim tarifi

Andrej Karpathy'nin *"A Recipe for Training Neural Networks"* (2019) yazısındaki yaklaşımın bu projeye uyarlanmış
hali. Ana fikir şu: sinir ağı eğitimi **sessizce** başarısız olur. Kod çökmez, sayılar makul görünür, ama yanlıştır.
Hızlı gitmenin yolu yavaş ve titiz gitmektir.

## 0. Bu projede "sessiz başarısızlık" neye benzer

Aşağıdakilerin hiçbiri hata vermez; hepsi yalnızca biraz daha kötü (ya da şüpheli derecede iyi) bir sayı üretir.

- **Önek uyumsuzluğu.** E5 ailesi `query:` ve `passage:` önekleri ister, BGE-M3 hiç önek istemez. Model değiştirip
  `prefixes()` güncellenmezse eğitim ve çıkarım farklı dağılımda çalışır.
- **Havuz karışıklığı.** Val metrikleri WARM havuzda (2.699 ilan, WARM IDCG), test metrikleri tam havuzda (3.000 ilan)
  hesaplanır. İkisi aynı ölçekte değildir. Val 0,70 ile test 0,66 karşılaştırması anlamsızdır.
- **Bayat paket.** Notebook çekirdeği paketten içe aktarır. Çekirdeği değiştirip paketi yeniden üretmezsen Colab eski
  kodu çalıştırır. Aynı Colab makinesinde eski `/content/data_bundle` silinmezse yeni zip hiç açılmaz.
- **Hassasiyet kayması.** Accelerate'in autocast sarmalayıcısı eğitimden sonra modelde kalır. `drop_amp_wrapper` bunu
  söker; bir sürüm değişikliği bu söküşü sessizce bozabilir. fp16 dışa aktarım ağırlıkları yuvarlar. Belirti: parite
  satırında Top-10 örtüşmesinin %100'ün belirgin altında olması.
- **Yanlış negatif.** Batch içindeki başka satırın pozitifi bu anchor için alakalı olabilir. Kıdem/yaka tuzağı
  negatiflerinin 1.730/4.883'ü grade 1'dir. Tanı satırı her çalıştırmada bunu yazdırır.
- **Test'e bakarak ayar.** Her çalıştırma testi yazdırır. Test sayısına bakıp yapılandırma değiştirmek, bir sonraki
  sayının test değil "ikinci val" olması demektir.
- **Kâhin tavanını unutmak.** Aşama 2'yi haftalarca ayarlayıp aday kümesinin tavanının zaten 87 olduğunu sonradan
  fark etmek.

Savunma, her aşamada bir **sağlama** yapmaktır: sonucunu önceden tahmin edebildiğin, tutmazsa durduğun bir kontrol.

## 1. Veriyle bir ol (GPU gerekmez, ≈1 saat)

Model kodundan önce veriye bak. Bu projede veri kural türevidir; kuralı içselleştirmeden model ayarlamak, neyi
ödüllendirdiğini bilmeden optimize etmektir.

**Yap:**
1. `python scripts/12_metric_ceilings.py` çalıştır, ardından `reports/metric_ceilings.json` dosyasını oku. Dört
   tavanı ezbere bil; özetleri `00` §3'te.
2. `scripts/05_build_dataset.py` içindeki `grade_matrix()` fonksiyonunu oku. Bu işlev 0–3 dereceyi deterministik
   olarak üretir.
   - **Grade ≥ 2** pratikte "aynı ISCO-4 birim grubu + yaka + |kıdem| ≤ 2" demektir. Testte aynı ISCO-4
     çiftlerinin %96,4'ü grade ≥ 2'dir.
   - **Grade 3** ek olarak aynı uzmanlık (`occ_key`), |kıdem| ≤ 1 ve aynı şehir ya da Remote ilan ister.
   - Bu yüzden Top-1 ve P@10 büyük ölçüde iki taraflı bir **ISCO-4 tanıma** problemidir. NDCG@10 ise ek olarak
     **uzmanlık + lokasyon** ayrımı ister.
3. 20 rastgele train CV'si seç. Her biri için en yüksek dereceli 5 ilanı ve BM25'in ilk 5'ini yan yana oku
   (`clean_cvs.parquet`, `clean_jobs.parquet`, `qrels_ground_truth.parquet`). Kendine sor:
   - Bir insan hangi ilanı seçerdi?
   - Kural neden farklı düşünüyor?
4. Vitrin CV'lerine bak (20951 baklava ustası, 10587 uçuş dispeçeri, 10511 Java geliştirici, 21029 Java stajyeri).
   - 20951'de grade-3 ilan **imkânsızdır**: aynı uzmanlıktaki iki ilan İzmir ve Hatay'dadır, aday Ankara'dadır.
   - 10587'nin train'deki tek kapsaması sentetik bir CV'dir.
5. `python scripts/11_local_lexical_ltr.py` ve `python scripts/13_lexical_lever_probe.py` çalıştır.
   - 11, BM25 + LambdaMART'ın test sonucunu verir (53,66 → 66,05).
   - 13, ISCO, kNN ve K=100 kaldıraçlarını **yalnız val** üzerinde, GroupKFold ile ölçer.
6. BM25'in Top-1 hatalarını sınıflandır. Hazırlıkta ölçülen tablo: hataların yarısı "sektör doğru ama meslek yanlış",
   %12,5'i yaka hatası; kıdem neredeyse hiç etkili değil. Bunlar ve ayrıntılar `04` §1'de.

**Çıkış ölçütü:** Bir CV ve bir ilan gösterildiğinde dereceyi tahmin edebiliyorsun. Hedeflerden hangisinin hangi
yapısal tavana çarptığını tek cümleyle söyleyebiliyorsun.

## 2. Uçtan uca iskelet + aptal baseline'lar

İskelet hazır: notebook. Bu adımın işi, iskeletin **doğru** olduğunu ve her parçanın beklenen sayıyı ürettiğini
kanıtlamaktır.

**Sağlamalar (ilk A100 çalıştırması):**
- **B1 (BM25) deterministiktir.** Test için tam olarak şunu üretmeli: `NDCG@10 53.66, MRR@10 74.03, P@10 38.37,
  P@10_capped 48.53, R@10 26.59, R@50 47.46, Top1 64.68, Top1_exact(grade3) 9.49`. Tek bir hane farklıysa veri veya
  çekirdek repo ile aynı değildir; **dur**.
- **Rastgele skor** (§6.1 tablosu) kademeli metrikte neredeyse sıfır vermeli.
- **Sıfır-atış BGE-M3** ("Ref: Zero-shot base bi-encoder") Aşama 1'in başlangıç noktasıdır. İnce ayarlı dense val'de
  bunu geçmiyorsa Aşama 1'de bir hata vardır; ayarlama yapma, hatayı bul.
- **Sayaçlar:**
  - `MNRL satırı: 4883 … CoSENT çifti: 9719`
  - `train anchor: 2786 | val sorgu: 457 | test sorgu: 461 (… 8)`
  - `LTR çerçeveleri: {'val': (22850, 38), 'test': (23050, 38)}`

  Biri farklıysa veri veya kod değişmiştir.
- **Parite satırı.** Paketten yeniden yüklenen model, notebook'un test sıralamalarıyla örtüşmeli. Örtüşme ≈ %100,
  Top-1 aynılığı ≈ %100 olmalı. Kuru çalıştırmada küçük modellerle %98,8 / %100 çıktı. Gerçek modelle %95'in altı
  araştırılmalı (`05` §A).

Tam sağlık tablosu `02` §3'tedir.

**Aptal baseline eklemek istersen:** "Popülerlik" sıralayıcısı, yani her ilanı train'de kaç CV'ye grade ≥ 2
olduğuna göre sıralamak, sorgudan bağımsız bir alt sınırdır. Bu baseline'ın üstünde kalmayan bir model gerçekte
sorguyu kullanmıyordur.

## 3. Önce aşırı öğren (kapasite var mı?)

Eğitilen model kendi eğitim verisini ezberleyemiyorsa sorun düzenlileştirme değil hatadır: veri hattı, kayıp,
örnekleyici ya da önek.

**Aşama 1 kapasite testi.** §7.2'den sonra yeni bir hücrede çalıştır:
```python
ids = [a for a in anchors[:300]]                       # eğitimde görülmüş train anchor'ları
Eq = encode(BI, cv_texts(ids, QP))
print("train (in-sample):", fmt(eval_scores((Eq @ E_jobs.T)[:, WARM_COLS], ids, cols=WARM_COLS, store=False)))
```
- Bu sayı val'deki `eval_val_graded_ndcg@10` değerinden belirgin biçimde yüksek olmalı.
- Train ve val **ikisi de** düşükse model öğrenemiyordur (yetersiz öğrenme). Bu durumda önce şunlara bak: LR, τ, gerçek
  adım sayısı (`stage1_training_log.csv` satır sayısı), kayıp eğrisi.
- Train yüksek, val düşükse genelleme sorunu vardır. 4. adıma geç.

**Aşama 2a kapasite testi.** Aynı mantıkla 200 train anchor'unun `cand_tr` listeleri CE ile yeniden sıralanır ve
NDCG ölçülür. `GradedRerankEvaluator` sınıfını `tr_anchor[:200]` ve `fused_tr[:200]` ile kur.

**LambdaMART.** Val üzerinde in-sample NDCG ile `ltr_cv_ndcg10` (kat dışı) arasındaki fark aşırı uyumun ölçüsüdür.

**Çıkış ölçütü:** Her öğrenen bileşen eğitim verisinde baseline'ı açıkça geçiyor.

## 4. Düzenlileştir (genelleme)

Sırayla, her seferinde tek bir değişiklik:
1. **Daha fazla gerçek sinyal.**
   - Aşama 1 yalnız 4.883 MNRL satırı (2.786 anchor) görür. Train qrels ise 94.522 WARM grade ≥ 2 çift sunar (3.689'u
     grade 3).
   - `build_mnrl_rows` içinde QRELS'ten ek pozitif satırlar eklenebilir; paketi yeniden üretmek gerekmez.
   - Grade-2 pozitifleri gürültülüdür (kör doğrulamada 43/90'ı 0–1 bulundu). Grade 3'e ağırlık ver ya da grade 2'yi
     sınırla.
2. **Sentetik verinin katkısını ölç.** `USE_SYNTHETIC=False` ile bir ablation yap. Sentetik CV'lerin 54'ü artık
   gereksiz hedeflere aittir (`05` §B).
3. **Daha iyi negatifler.**
   - İkinci tur zor negatif madenciliği: ince ayarlı modelle yeniden madenle, qrels ile süz (ANCE / RocketQA).
   - Ya da kayıp içinde tam bir yanlış-negatif maskesi kullan: qrels tam olduğu için grade ≥ 2 olan batch içi
     negatiflerin logit'i −∞ yapılabilir.
4. **Erken durdurma.** Eğri `stage1_training_log.csv` / `stage2a_training_log.csv` dosyalarındadır.
   - En iyi değerlendirme ilk değerlendirmeyse LR çok yüksek ya da epoch sayısı fazladır.
   - Son değerlendirmede hâlâ yükseliyorsa epoch ekle.
   - CE için 4 değerlendirme ve sabır 3 ayarı erken durdurmayı fiilen etkisiz kılar.

**Çıkış ölçütü:** Val eğrisi tepe yapıp duruyor; train ile val arasındaki fark açıklanabilir.

## 5. Ayarla (yalnız val)

- **Önce tavanı ölç, sonra ayarla.**
  - §9.1, hibrit adayların val tavanını yazdırır. Bu tavan 92'nin altındaysa Aşama 2 ayarı boşa gider.
  - O durumda `N_CANDIDATES`'ı artır (50 → 100 hesap maliyetini kabaca ikiye katlar), RRF `k` ve kanal ağırlıklarını
    val R@50/R@100'e göre ayarla (`core.rrf_scores` ağırlık alır), ya da dense geri getirmeyi güçlendir.
- **Önemli düğmeler:** `BI_LR`, τ (`TEMP_GRID`), `BI_EPOCHS`, `N_CANDIDATES`, `CE_DOCS_PER_QUERY`, `CE_LR`, `CE_MAX_LEN`.
  Gerisi ikinci derecedendir. Izgara yerine rastgele arama kullan (Bergstra & Bengio, 2012).
- **τ taraması bilgi vermeyebilir.** Tarama yarım veriyle, 1 epoch, ≈38 adım ve CoSENT olmadan yapılır. τ'yu sabitlemek
  ya da taramayı uzatmak meşru bir seçimdir.
- **Her ayarlama kararı yalnız şu sinyallerle verilir:**
  - `eval_val_graded_ndcg@10` (Aşama 1 ve 2a log'ları);
  - `ltr_cv_ndcg10`;
  - §9.1 val tavanı;
  - `scripts/13` tarzı val GroupKFold ölçümleri.

  Hepsi `run_summary.json` → `val_sinyalleri` / `tani` altında toplanır.

## 6. Son damlaları sık

Ancak 1–5 bittiyse şunlar sırayla düşünülebilir:
- topluluk (birden çok CE tohumu veya modeli, LambdaMART'a ayrı öznitelik olarak);
- daha büyük modeller (Qwen3-Reranker, bge-reranker-v2-gemma, Qwen3-Embedding);
- CE → bi-encoder damıtma (Margin-MSE);
- daha uzun bağlam (`MAX_SEQ_LEN` / `CE_MAX_LEN` 768–1024, önce kırpılma oranını §9.1'de ölç);
- çapraz uydurulmuş LTR (train sorgularında, kat dışı dense/CE öznitelikleriyle).

Maliyet ve kanıt düzeyleri `04` §3'te. Her biri bir tam çalıştırma ya da daha pahalısıdır.

## İlkeler

1. **Tek seferde tek değişiklik.** İki şeyi birden değiştirirsen hangisinin işe yaradığını bilemezsin.
2. **Önce hipotez, sonra deney.** `06_DENEY_GUNLUGU.md`'ye ne beklediğini **çalıştırmadan önce** yaz.
3. **Test mühürlü zarftır.** Her çalıştırma test yazdırır; sen yalnız kaydet. Kararı val'e göre ver. Raporda
   testin kaç kez okunduğunu (`run_summary.json` → `test_degerlendirme_sayisi`, çalıştırma sayısı) dürüstçe yaz.
4. **Tavanla konuş.** Her sonucu kâhin tavanının yanında ver: "NDCG@10 = X (tavan 100; ikili kusursuz 91,36;
   aday tavanı Y)."
5. **Kanıt yoksa iddia yok.** Her sayı bir dosyaya (JSON/CSV) dayanmalı. "Muhtemelen" ile başlayan cümleler deney
   önerisidir, sonuç değildir.
6. **Kahraman olma.** Büyük model ve karmaşık kayıplar en sona kalır. Bu verinin yapısı (ISCO-4, uzmanlık, lokasyon)
   basit özniteliklerle çok şey kazandırıyor; bunu `scripts/13`'te gör.
