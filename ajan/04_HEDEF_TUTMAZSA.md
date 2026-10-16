# Hedef tutmazsa

İlk refleks şu olmalı: **önce ölç, sonra değiştir.** Bu belgenin sırası da bu: önce tavanlar, sonra teşhis ağacı,
sonra kaldıraçlar, en sonda dürüst rapor. Kaldıraçlar öneri listesidir; hangisini, hangi sırayla deneyeceğin
senin kararındır.

## 0. Hangi hedef yapısal olarak nerede

`reports/metric_ceilings.json` dosyasından alınan değerler (test, tam havuz, alakalı = grade ≥ 2):

| Hedef | Kâhin | İkili kusursuz¹ | BM25 ilk-50 → kâhin² | BM25 ilk-100 → kâhin² | Yorum |
|---|---|---|---|---|---|
| Top-1 > 88 | 100 | 100 | 96,91 | 99,56 | Ulaşılabilir. Pratikte iki taraflı **ISCO-4 tanıma** problemi |
| NDCG@10 > 92 | 100 | **91,36** | **86,95** | 93,37 | Grade 3'ü 2'den ayırmak **ve** iyi adaylar şart |
| P@10 > 85 | **83,38** | 83,38 | 65,76 | 73,84 | **Ulaşılamaz** (sorguların %34,5'inin alakalısı < 10) |
| P@10_capped > 85 | 100 | 100 | 79,12 | 88,11 | Ulaşılabilir; `TARGETS`'ta zaten var |

¹ Tüm grade ≥ 2 ilanlar başta, kendi aralarında rastgele; geri kalanlar doğru sırada.
² BM25'in ilk K adayı gerçek dereceye göre yeniden sıralanır = bu adaylarla Aşama 2'nin üst sınırı.

Val'de aynı değerler: kâhin P@10 85,08, ikili kusursuz 91,96, BM25 ilk-50 → kâhin 83,15, ilk-100 → 89,71.

**Bundan çıkanlar:**
1. NDCG@10 > 92 için model iki şeyi aynı anda yapmalı:
   - **Uzmanlık** (`occ_key`): ISCO-4'ün altındaki kanonik meslek.
   - **Lokasyon:** aynı şehir ya da Remote ilan. Aynı uzmanlık/yaka/kıdem çiftlerinin %69,9'u yalnız şehir
     yüzünden grade 2'de kalır.

   Lokasyon metinden kusursuz okunur (`loc_ok`, `same_city`, `job_remote`). Uzmanlık ise zordur: test sorgularının
   ≈%21,5'inin uzmanlığı train'de hiç geçmez.
2. Aşama 2'nin yapabileceği, aday kümesinin tavanıyla sınırlıdır. BM25 kalitesindeki 50 aday ≈87'de takılır.
   Hibrit adayların tavanını notebook §9.1 **val** üzerinde yazdırır. Önce ona bak.
3. P@10 > 85'i kovalamak boşunadır. Doğru davranış: P@10'u tavanıyla birlikte raporlamak (örn. "P@10 = X; kâhin
   83,38; kâhine oran X/83,38"), P@10_capped'i vermek ve hedefin yeniden ifadesini **kullanıcıya sormak**.

## 1. Teşhis ağacı (yukarıdan aşağı; bir düğümde sorun varsa aşağı inme)

1. **Sağlamalar geçiyor mu?** B1 = 53,66 …, `Sızıntı koruyucuları: TAMAM`, sayaçlar, `sorgu = 453`, parite ≈ %100
   (`02` §3).
   - Hayır → hata ayıkla; hiçbir sayıyı yorumlama.
2. **Aşama 1 öğreniyor mu?** İnce ayarlı dense'in val NDCG'si (`stage1_training_log.csv`) sıfır-atış BGE-M3'ten
   yüksek mi? Kapasite testi (`01` §3) train'de yüksek sonuç veriyor mu?
   - Hayır → şunlara bak: önekler (`prefixes()`), LR, τ, gerçek adım sayısı, örnekleyici (batch boyları),
     yanlış-negatif tanısı, `enable_grad_ckpt` / `drop_amp_wrapper`.
3. **Aday tavanı yeterli mi?** §9.1 val NDCG@10 tavanı, hedefin rahatça üstünde mi? (Kaba kural: ≥ 95.)
   - Hayır → **Aşama 2'ye dokunma.** Sıra şu: `N_CANDIDATES` (50 → 100), RRF ağırlık ve `k` ayarı (val R@50/R@100),
     dense geri getirmeyi güçlendirmek (§3 K1–K6).
4. **Cross-encoder katkı veriyor mu?** `ablation_component_tests.csv` → "Cross-encoder (Hybrid+CE vs S1)" satırı
   (kayıt) ve CE val eğrisi (karar).
   - Katkı küçük ya da negatifse §3 K7–K10.
5. **LambdaMART katkı veriyor mu, aşırı uyuyor mu?**
   - Bileşen satırına bak: "LambdaMART (S2 vs Hybrid+CE)".
   - `ltr_cv_ndcg10` ile val in-sample NDCG'yi karşılaştır.
   - Öznitelik önemlerine (gain/SHAP) bak.
   - Sonra §3 K2–K4 ve K11.
6. **Top-1 iyi ama NDCG düşük mü?** Bu, grade-3 ayrımının zayıf olduğunu gösterir.
   - Kovaları incele: grade-3'ü olan sorgular; `occ_key`'i train'de görülmüş ve görülmemiş sorgular; Remote ve şehir.
   - `loc_ok` özniteliğinin LTR'deki önemine bak.
7. **Kovaya özgü bir zayıflık var mı?** Örnek kovalar: mavi yaka, düşük `n_rel`'li sektörler (LOGISTICS_DRIVER,
   HEALTHCARE, AGRICULTURE…), görülmemiş uzmanlıklar.
   - Hedefli veri: bu uzmanlıklar için ek pozitif satır ya da sentetik veri. Sentetik veri yalnız ilan tarafındaki
     WARM bilgiden üretilir ve test'e bakılarak hedef seçilmez.

Kovaları **val** üzerinde kur. `per_query_test.csv` yalnız kayıt içindir. Val için sorgu bazlı sonuçlar notebook
içinde `RESULTS` yerine val değerlendiricilerinden ya da `scripts/13` tarzı kat dışı tahminlerden alınır.

## 2. GPU'suz ölçülenler (yalnız val, leksikal hat)

`scripts/13_lexical_lever_probe.py` → `reports/lexical_lever_probe_val.json`:
- Aday: BM25 ilk-K. Dense ve CE öznitelikleri 0.
- Val sorgularında 5 katlı GroupKFold (grup = `group_id`), kat dışı sıralama, tam havuz qrels.
- p-değeri: "base, K=50"ye karşı eşleştirilmiş randomizasyon.
- **Test kullanılmadı.**

| Varyant | NDCG@10 [%95 GA] | Top-1 | P@10 | P@10_capped | R@50 | p (NDCG) | p (Top-1) |
|---|---|---|---|---|---|---|---|
| K=50 base | 60,44 [57,9; 63,2] | 68,93 | 42,31 | 49,16 | 42,20 | — | — |
| K=50 +isco | 65,15 [62,5; 67,9] | 73,70 | 48,64 | 55,39 | 42,20 | < 0,001 | 0,008 |
| K=50 +knn | 65,34 [62,7; 68,2] | 71,88 | 48,96 | 55,65 | 42,20 | < 0,001 | 0,190 |
| K=50 +knn+isco | 65,80 [63,0; 68,6] | 72,34 | 49,80 | 56,37 | 42,20 | < 0,001 | 0,119 |
| K=100 base | 61,56 [59,0; 64,3] | 69,84 | 42,79 | 49,38 | 48,15 | 0,013 | 0,596 |
| K=100 +isco | 67,00 [64,4; 69,7] | 70,98 | 51,16 | 58,25 | 50,30 | < 0,001 | 0,299 |
| K=100 +knn | 67,36 [64,6; 70,2] | 70,29 | 51,11 | 57,97 | 48,28 | < 0,001 | 0,596 |
| K=100 +knn+isco | 67,56 [64,8; 70,4] | 70,75 | 51,52 | 58,38 | 48,75 | < 0,001 | 0,448 |

(val, 441 sorgu; ISCO sınıflandırıcısının val CV'lerinde ilk-1 doğruluğu %59,1. Ölçüm süresi ≈16 dk, 4 CPU.)

**Özet:** ISCO-4 özniteliği hem NDCG@10'u (+4,7) hem Top-1'i (+4,8; p = 0,008) anlamlı biçimde artırır. kNN, NDCG'yi benzer ölçüde artırır ama Top-1'de anlamlı değildir; ikisi birlikte ISCO'ya küçük bir şey ekler. K=100 NDCG'yi artırır (aday tavanı yükselir), Top-1'i anlamlı biçimde artırmaz. Mutlak düzey (≈65–68) hedefin (92) çok altındadır: leksikal hat tek başına yetmez; nöral aşamalar şarttır.

**Okuma notları:**
- Bu hat nöral değildir. Dense ve CE öznitelikleri eklenince kazançlar küçülebilir ya da başka bir yere kayabilir.
  Kaldıracı A100 hattına taşıdıktan sonra val'de yeniden doğrula.
- **ISCO-4 özniteliği** şöyle üretilir: train CV metninden `gold_isco4`'ü tahmin eden bir LinearSVC eğitilir; bu
  sınıflandırıcı hem CV'ye hem ilan metnine uygulanır; özniteliği olasılık vektörlerinin iç çarpımı ve ilk-1
  eşleşmesi oluşturur. İlan tarafının altın etiketi kullanılmaz. Bu, `AttributeModels`'ın sektör/yaka için
  kullandığı yolla aynıdır.
- **kNN etiket aktarımı** şöyle üretilir: en yakın 30 train CV'sinin o ilana verdiği dereceler alınır; cold ilanlar
  0'dır. Cold, yani yeni ilanlara hiç sinyal vermez. Ürün davranışını değiştirdiği için kullanmadan önce kullanıcıya
  sor. Ayrıca çıkarım paketine train dereceleri (2.835 × 3.000) eklenmesi gerekir.

## 3. Kaldıraçlar (kanıt düzeyi ve maliyete göre)

Kanıt düzeyleri: **V** = bu repoda val'de ölçüldü (leksikal hat); **L** = literatürde güçlü kanıt, bu veride
ölçülmedi; **?** = bilinmiyor.

| # | Kaldıraç | Nerede | Mekanizma | Kanıt | Maliyet | Risk |
|---|---|---|---|---|---|---|
| K1 | `N_CANDIDATES` 50 → 100 | `CFG` | Aday tavanını yükseltir (BM25'te 86,95 → 93,37) | V + tavan | CE değerlendirme/çıkarım ×2 | Düşük; çıkarımda top_k sınırı da değişir |
| K2 | ISCO-4 eşleşme özniteliği | `core.AttributeModels` + `FEATURE_COLUMNS` + `build_pair_features` | grade ≥ 2 ≈ aynı ISCO-4 | V | CPU'da saniyeler; çekirdek ve paket değişir | Düşük (yalnız train etiketleri) |
| K3 | Uzmanlık (`occ_key`) düzeyinde öznitelik: unvan/uzmanlık sınıflandırıcısı ya da en yakın train unvanı | aynı | grade 3 = aynı `occ_key` (+ lokasyon) | ? | Saniyeler–dakikalar | 1.349 sınıf, seyrek; görülmemiş uzmanlıklar |
| K4 | kNN etiket aktarımı (öznitelik ya da 3. RRF kanalı) | çekirdek + paket | Benzer CV'lerin dereceleri | V | Saniyeler; paket büyür | **Kullanıcı kararı** (cold ilanlara karşı yanlılık) |
| K5 | RRF ağırlık / `k` ayarı | `hybrid()` → `core.rrf_scores(weights=…)` | Güçlü kanala daha çok ağırlık | ? | Saniyeler | Düşük |
| K6 | Aşama 1: daha fazla pozitif (QRELS'ten), qrels ile tam yanlış-negatif maskesi, ikinci tur zor negatif (ANCE/RocketQA), CE → bi damıtma (Margin-MSE) | `build_mnrl_rows`, kayıp | Daha çok ve daha temiz sinyal, daha iyi geri getirme | L | 1–2× Aşama 1 | Grade-2 pozitifleri gürültülü |
| K7 | CE eğitim listelerini dağılım uyumlu kur: base/BM25 adayları ya da çapraz uydurulmuş BI | §9 hücresi | Train listeleri val/test kadar zor olur (`05` B.6) | ? | 0 ya da K× Aşama 1 | Düşük |
| K8 | `CE_DOCS_PER_QUERY` 16 → 32; `LambdaLoss(k=10)`; `CE_EPOCHS` 2 → 3; `CE_LR` taraması | `CFG`, §9 | Daha çok negatif, ilk-10'a odak | ? | ≈2× CE | Bellek (`02` §4) |
| K9 | `CE_MAX_LEN` 768–1024 / akıllı sıkıştırma (Pozisyon/Kıdem/Konum/Yetkinlikler önde) | `CFG`, `core.compact_*_for_ce` | Kırpılma kaybını azaltır (önce §9.1'de ölç) | ? | 2–4× CE | Bellek |
| K10 | Daha büyük reranker (bge-reranker-v2-gemma, Qwen3-Reranker 4B/8B) — sıfır-atış skoru LTR özniteliği olarak ya da LoRA ile ince ayar | §9, çekirdek | Daha güçlü çapraz dikkat | L (genel kıyaslamalar) | Saatler; 40 GB'ta LoRA gerekebilir | **Lisans kontrolü**; peft sabit değil |
| K11 | LTR'yi çapraz uydurulmuş train sorgularında eğit (kat dışı dense/CE öznitelikleri) | §10 + Aşama 1/2a K-katlı | 457 val sorgusundan çok daha fazla LTR verisi | ? (leksikal hatta keşif amaçlı olumlu) | ≈5× Aşama 1+2a | Kat dışı olmayan train öznitelikleri **yasak** (sızıntı) |
| K12 | Daha güçlü embedder (Qwen3-Embedding, multilingual-e5-large-instruct, Türkçe modeller) | `BI_ENCODER_BASE` + `prefixes()` | Daha iyi geri getirme | L | Tam çalıştırma | Önek/talimat biçimi; lisans |
| K13 | BGE-M3 seyrek / çok-vektör modları | FlagEmbedding (sabit değil) | Leksikal + çok-vektör sinyali | L (M3 makalesi) | Orta | Yeni bağımlılık |
| K14 | Topluluk / test-zamanı çoğaltma (başlık-yalnız, beceri-yalnız sorgu görünümleri) | §8/§10 | Varyansı düşürür | ? | Üye sayısıyla doğrusal | Düşük |
| K15 | Dış LLM ile liste-bazlı yeniden sıralama (RankGPT) | yeni | Güçlü sıralayıcı | L | API maliyeti | **Kullanıcı kararı** (gizlilik). Etiketler de LLM kökenli olduğundan dairesellik riski var; ayrı bir sistem olarak raporla |

Model adlarını ve lisanslarını kullanmadan önce model kartında doğrula. Hazırlık sırasında huggingface.co'ya
erişilemedi; parametre sayıları ve lisanslar ikincil kaynaklardan alındı.

## 4. Kestirme yollar: yapma

Bunlar sayıyı yükseltir ama modeli iyileştirmez. Hepsi metodoloji ihlalidir. Gerekçeli bir yeniden tanım ancak
kullanıcı onayıyla ve eski ve yeni sayılar ayrı raporlanarak yapılabilir.

- **`REL_THRESHOLD=1`.** Aynı BM25 sıralaması P@10'da 38,37 → 63,91 ve Top-1'de 64,68 → 82,39 olur. Model hiç
  değişmez; tanım değişir.
- **Alakalısı < 10 olan sorguları atmak** (P@10 kâhini 100 olur) ya da başka bir alt-küme seçmek.
- **`grade_matrix()` kurallarını hedefi tutturacak biçimde değiştirmek.** Örnek: lokasyonu grade 3'ten çıkarmak,
  grade-3'ü olan sorguların payını %33,6'dan %83,1'e çıkarır.
- **İlan tarafının altın etiketlerini öznitelik yapmak.** Kuralın yarısını doğrudan gözlemlenebilir kılar.
- **Test sayısına bakıp yapılandırma seçmek** ya da birden çok çalıştırmadan testte en iyisini raporlamak.
- **Cold ilanları ya da test CV'lerini herhangi bir eğitim veya seçim sinyaline sokmak.**

## 5. Hedef yine tutmazsa: dürüst rapor

Bu bir başarısızlık değil, bir bulgudur, yeter ki ölçülmüş ve açıklanmış olsun. Raporda şunlar bulunmalı:

1. Elde edilen değerler (test, tek seferlik okuma) ve %95 GA'ları.
2. Her hedefin tavanı:
   - kâhin;
   - ikili kusursuz;
   - aday tavanı (§9.1; val değeri karar için, test değeri rapor için);
   - tavana oran.
3. Bileşen katkıları ve p-değerleri (`ablation_component_tests.csv`).
4. Denenen kaldıraçlar ve val etkileri (`06` günlüğünden tablo), toplam tam çalıştırma sayısı.
5. Hata analizi: hangi kovalar kaybettiriyor, bunların ne kadarı etiket gürültüsü (kör doğrulama: kural grade-3'lerin
   yarısı yargıçça 2).
6. Kalan hipotezler ve maliyetleri.
7. Kullanıcıya **soru**, cevabı kullanıcı verir. Örnek:
   > "P@10 > 85 bu qrels tanımıyla imkânsız (kâhin 83,38). Hedefi P@10_capped > 85 ya da 'kâhine oranla
   > P@10 > %X' olarak yeniden ifade etmeyi öneriyorum. Onaylıyor musunuz?"
