# Hedef tutmazsa

İlk refleks şu olmalı: **önce ölç, sonra değiştir.** Bu belgenin sırası da bu: önce tavanlar, sonra teşhis ağacı,
sonra kaldıraçlar, en sonda dürüst rapor. Kaldıraçlar öneri listesidir; hangisini, hangi sırayla deneyeceğin
senin kararındır.

Her karar yalnız **val, WARM havuz** sinyaliyle verilir: Aşama 1/2a log'ları (`eval_val_*`),
`val_zero_shot_base(WARM)`, `ltr_cv_ndcg10`, §9.1 `aday_tavani_val(WARM)`, §10.1 `VAL_TBL` / `VAL_COMP` /
`per_query_val.csv`, `scripts/13` (`03` §A). Notebook'un yazdırdığı bütün test satırları ve tabloları (§4–§12 ve §14; ör. B1/B2/B3, §8 Recall, §11 ABL/COMP, §11.1, §12), `per_query_test.csv` ve
`run_summary.json` → `hedefler` / `ana_ablation` yalnız kayıt ve son rapor içindir. Ölçek: log'lar, `ltr_cv_ndcg10`
ve `per_query_val.csv` 0–1'dir; yazdırılan sözlükler ve tablolar yüzdedir.

## 0. Hangi hedef yapısal olarak nerede

`reports/metric_ceilings.json` dosyasından alınan değerler (test, tam havuz, alakalı = grade ≥ 2). Hedefler test
üzerinde tanımlı olduğu için bu tablo ulaşılabilirlik ve son rapor içindir; karar için alttaki val değerlerine bak.

| Hedef | Kâhin | İkili kusursuz¹ | BM25 ilk-50 → kâhin² | BM25 ilk-100 → kâhin² | Yorum |
|---|---|---|---|---|---|
| Top-1 > 88 | 100 | 100 | 96,91 | 99,56 | Ulaşılabilir. Pratikte iki taraflı **ISCO-4 tanıma** problemi |
| NDCG@10 > 92 | 100 | **91,36** | **86,95** | 93,37 | Grade 3'ü 2'den ayırmak **ve** iyi adaylar şart |
| P@10 > 85 | **83,38** | 83,38 | 65,76 | 73,84 | **Ulaşılamaz** (sorguların %34,5'inin alakalısı < 10) |
| P@10_capped > 85 | 100 | 100 | 79,12 | 88,11 | Ulaşılabilir; `TARGETS`'ta zaten var |

¹ Tüm grade ≥ 2 ilanlar başta, kendi aralarında rastgele; geri kalanlar doğru sırada.
² BM25'in ilk K adayı gerçek dereceye göre yeniden sıralanır = bu adaylarla Aşama 2'nin üst sınırı.

Val'de, karar ölçeğinde (WARM havuz): kâhin P@10 83,22; BM25 ilk-50 / ilk-100 / ilk-200 → kâhin NDCG@10
84,14 / 90,67 / 94,58 (`val|WARM|bm25_ilk{K}_kahin_yeniden_siralama`). Tam havuzda: kâhin P@10 85,08, ikili kusursuz
91,96 (WARM sürümü yok), BM25 ilk-50 / ilk-100 → kâhin 83,15 / 89,71.

**Bundan çıkanlar:**
1. NDCG@10 > 92 için model iki şeyi aynı anda yapmalı:
   - **Uzmanlık** (`occ_key`): ISCO-4'ün altındaki kanonik meslek.
   - **Lokasyon:** aynı şehir ya da Remote ilan. Val'de aynı uzmanlık/yaka/kıdem çiftlerinin %71,3'ü yalnız lokasyon
     yüzünden grade 2'de kalır (`metric_ceilings.json` → `grade3_lokasyon_bagimliligi`).

   Lokasyon metinden kusursuz okunur (`loc_ok`, `same_city`, `job_remote`). Uzmanlık ise zordur: val sorgularının
   %23,9'unun uzmanlığı train'de (sentetik CV'ler dahil) hiç geçmez (rapor dosyasına kayıtlı değil;
   `data_bundle/clean_cvs.parquet`'ten yeniden hesaplanır: val `gold_occ_key`'lerinden train'de hiç olmayanların payı).
2. Aşama 2'nin yapabileceği, aday kümesinin tavanıyla sınırlıdır. BM25 kalitesindeki 50 aday val'de (WARM) ≈84'te
   takılır. Hibrit adayların tavanını notebook §9.1 **val, WARM havuzda** yazdırır (`tani` → `aday_tavani_val(WARM)`).
   Önce ona bak; eşik §1 düğüm 3'te.
3. P@10 > 85'i kovalamak boşunadır. Doğru davranış: P@10'u tavanıyla birlikte raporlamak (örn. "P@10 = X; kâhin
   83,38; kâhine oran X/83,38"), P@10_capped'i vermek ve hedefin yeniden ifadesini **kullanıcıya sormak**.

## 1. Teşhis ağacı (yukarıdan aşağı; bir düğümde sorun varsa aşağı inme)

1. **Sağlamalar geçiyor mu?** (`02` §3): §2'deki `paket parmak izi` repodaki
   `sha256sum data_bundle/metadata.json match_candidate_sota.py | cut -c1-12` ile aynı, `Sızıntı koruyucuları: TAMAM`,
   sayaçlar, B1 = 53,66 … (test; yalnız sağlama, BM25 deterministiktir), §10.1'de `sorgu = 438`, §11'de `sorgu = 453`,
   parite ≈ %100.
   - Hayır → hata ayıkla; hiçbir sayıyı yorumlama.
2. **Aşama 1 öğreniyor mu?** `stage1_training_log.csv` → en iyi `eval_val_graded_ndcg@10` (`bi_val_en_iyi`, 0–1) × 100,
   §7'deki `Zero-shot base bi-encoder (VAL, WARM havuz — …)` NDCG@10'undan (`val_zero_shot_base(WARM)`) yüksek mi?
   §10.1 "İnce ayar (val)" ΔNDCG@10 pozitif mi? Kapasite testi (`01` §3) train'de yüksek sonuç veriyor mu? §7'nin ilk
   `Zero-shot base bi-encoder:` satırı testtir; bu kontrolde kullanılmaz.
   - Hayır → şunlara bak: önekler (`prefixes()`), LR, τ, gerçek adım sayısı (log'daki `step` sütununun en büyüğü;
     satır sayısı değil), örnekleyici (batch boyları), yanlış-negatif tanısı, `enable_grad_ckpt` / `drop_amp_wrapper`.
3. **Aday tavanı yeterli mi?** §9.1 `Aday kümesi tavanı (VAL, WARM havuz, …)` → NDCG@10 (`tani` →
   `aday_tavani_val(WARM)`). Tek kural:
   - **< 92** → **Aşama 2'ye dokunma.** Sıra şu: Aşama 1, `N_CANDIDATES` (50 → 100), RRF ağırlık ve `k` ayarı, dense
     geri getirmeyi güçlendirmek (§3 K1, K5, K6, K12, K13). Ölçüt val WARM R@50/R@100: §10.1 "S1" `R@50` (= §9.1
     `R@50`); R@100 için `per_query_val.csv` → S1 satırlarının ortalaması.
   - **92–95** → gri bölge: Aşama 1 / aday iyileştirmesi muhtemelen hâlâ en verimlisidir.
   - **≥ 95** → Aşama 2 işi (düğüm 4–5) karşılık verebilir.
   - Kıyas tabanı: val WARM BM25 ilk-50 84,14 / ilk-100 90,67 (§9.1 bunları `Kıyas (val, WARM, kâhin NDCG@10): …`
     satırında yazdırır). Val aday tavanları test tavanlarından birkaç puan düşüktür
     (BM25 ilk-100, WARM: val 90,67, test 93,47): eşikler testteki yapısal sınır değil, val üzerinde temkinli karar
     kurallarıdır. `aday_tavani_test(tam_havuz,yalniz_rapor)` yalnız rapor içindir.
4. **Cross-encoder katkı veriyor mu?** Karar: §10.1 `VAL_COMP` → "Cross-encoder (val)" (ΔNDCG@10, ΔTop1, p) ve CE val
   eğrisi (`stage2a_training_log.csv` → `eval_val_graded_ndcg@10`; `ce_val_en_iyi`). Test satırı "Cross-encoder
   (Hybrid+CE vs S1)" (`ablation_component_tests.csv`) yalnız kayıttır.
   - Katkı küçük ya da negatifse §3 K7–K10.
5. **LambdaMART katkı veriyor mu, aşırı uyuyor mu?** (karar yalnız val)
   - Katkı: §10.1 `VAL_COMP` → "LambdaMART (val, kat dışı)". Test satırı "LambdaMART (S2 vs Hybrid+CE)" yalnız kayıttır.
   - Aşırı uyum: `ltr_cv_ndcg10` (`val_sinyalleri` → `ltr_cv_ndcg10(val,WARM)`, 0–1) × 100 ile §10.1 "S2 … (kat dışı)"
     NDCG@10 yakın olmalı (aynı katlar ve parametreler; CV değeri 40 denemenin en iyisi olduğu için biraz iyimserdir).
     Val in-sample NDCG'yi notebook yazdırmaz; gerekirse 07 ile §10'a `ltr_rank_eval(RANKER, ltr_tr, None, None)` ekle
     ve `ltr_cv_ndcg10` ile kıyasla. Büyük fark aşırı uyumdur.
   - Öznitelik önemi: gain (`RANKER` val WARM satırlarında eğitilir) karar için kullanılabilir, ama
     `ltr_importance.png`'den okunmaz: figürün iki paneli de 35 öznitelikten `LTR["test"]` üzerindeki ort. |SHAP|
     sırasına göre ilk 20'yi gösterir (`imp.sort_values("mean_abs_shap")` → `head(20)`); figür yalnız kayıttır. Tüm
     özniteliklerin gain'i paketteki modelden alınır:
     `lgb.Booster(model_file="export_model_bundle/lambdamart.txt").feature_importance("gain")` (adlar:
     `feature_name()`). SHAP gerekiyorsa 07 ile `ltr_tr` üzerinde hesapla.
   - Sonra §3 K2–K4 ve K11.
6. **Top-1 iyi ama NDCG düşük mü?** (val: §10.1 `VAL_TBL` → "S2 … (kat dışı)") Bu, grade-3 ayrımının zayıf olduğunu
   gösterir.
   - Kovaları val'de incele: grade-3'ü olan sorgular; `occ_key`'i train'de görülmüş ve görülmemiş sorgular; Remote ve
     şehir.
   - `loc_ok` özniteliğinin LTR'deki gain'ine bak (`lambdamart.txt`'ten; png'deki test SHAP sıralı ilk 20'den değil).
7. **Kovaya özgü bir zayıflık var mı?** (val) Örnek kovalar: mavi yaka (val'de alakalı sayısının medyanı 8, beyaz
   yakada 37,5; `metric_ceilings.json` → `alakali_dagilimi`), düşük `n_rel`'li sektörler, görülmemiş uzmanlıklar.
   Bazı sektörlerde val'de yalnız 3–5 CV var (`faz2_veri_ozeti.json` → `split.val.kategori_dagilimi`): bu kovalarda
   farkı GA ile oku.
   - Hedefli veri: bu uzmanlıklar için ek pozitif satır ya da sentetik veri. Yeni sentetik CV, LLM ile üretim ve kör
     doğrulama ister (`scripts/06_synthetic_cvs.py targets` → `annotation/synthetic/{gen,verify}/` → `06 … merge` →
     `05`; `targets` kayıtlı `specs/`'i silip yeniden yazar). Bu bir **kullanıcı kararıdır** (`00` §4, maliyet): önce
     sor. Üretim yalnız ilan tarafındaki WARM bilgiyi kullanır; hedefler teste bakılarak seçilmez.

Kovaları **val** üzerinde kur: `per_query_val.csv` (§10.1; her `sistem` için sorgu başına metrikler ve `n_rel`, WARM
havuz, 0–1) + CV altın sütunları ve qrels (`clean_cvs.parquet` → `gold_*`, `qrels_ground_truth.parquet`; değerlendirme
tarafında serbest, öznitelik yapılmaz). `per_query_test.csv`, §11.1 ve §12 yalnız kayıt ve son rapor içindir; bir
sonraki kaldıraç onlara bakılarak seçilmez. §10.1 sayıları hafifçe iyimserdir (val erken durdurma, checkpoint ve LTR
hiperparametre seçiminde kullanıldı): değişiklikleri birbiriyle kıyaslamak içindir, yansız kestirim değildir. İki
çalıştırmayı aynı `sistem`'in `per_query_val.csv` satırlarını `qid` üzerinden eşleştirip
`core.paired_randomization_test` ile kıyasla. Drive'daki `MyDrive/cv_ilan_sota/` her çalıştırmada üzerine yazılır:
önceki dosyalar bir sonraki çalıştırmadan önce `runs/<tarih>_<etiket>/` altına kopyalanmış olmalı. `scripts/13` yalnız
özet yazar; leksikal kovalar gerekirse önce ona sorgu bazlı çıktı ekle.

## 2. GPU'suz ölçülenler (yalnız val, leksikal hat)

`scripts/13_lexical_lever_probe.py` → `reports/lexical_lever_probe_val.json` (taze klonda önce repo kökünde
`unzip -o data_bundle.zip`; `data_bundle/` git'te izlenmez):
- Aday: BM25 ilk-K; cold adaylar elenir (WARM aday sayısı K'dan az olabilir). Dense ve CE öznitelikleri 0.
- Val sorgularında 5 katlı GroupKFold (grup = `group_id`), kat dışı sıralama. Değerlendirme WARM havuzda: cold
  ilanlar ne sıralamada ne qrels'te (notebook'un seçim değerlendiricileriyle aynı ölçek).
- p-değeri: "base, K=50"ye karşı eşleştirilmiş randomizasyon.
- **Test kullanılmadı.**

| Varyant | NDCG@10 [%95 GA] | Top-1 | P@10 | P@10_capped | R@50 | p (NDCG) | p (Top-1) |
|---|---|---|---|---|---|---|---|
| K=50 base | 60,49 [58,2; 63,0] | 66,89 | 41,35 | 49,33 | 44,86 | — | — |
| K=50 +isco | 65,27 [62,7; 67,8] | 71,92 | 47,51 | 55,48 | 44,86 | < 0,001 | 0,006 |
| K=50 +knn | 67,09 [64,6; 69,7] | 72,37 | 49,22 | 57,78 | 44,86 | < 0,001 | 0,010 |
| K=50 +knn+isco | 67,14 [64,6; 69,9] | 72,15 | 49,63 | 57,74 | 44,86 | < 0,001 | 0,012 |
| K=100 base | 61,44 [59,2; 63,9] | 68,72 | 41,53 | 49,06 | 49,74 | 0,038 | 0,280 |
| K=100 +isco | 67,42 [65,0; 69,9] | 70,78 | 50,14 | 58,26 | 51,85 | < 0,001 | 0,037 |
| K=100 +knn | 69,10 [66,4; 71,7] | 70,55 | 51,32 | 60,26 | 52,19 | < 0,001 | 0,090 |
| K=100 +knn+isco | 69,29 [66,6; 72,0] | 71,00 | 51,51 | 60,41 | 52,46 | < 0,001 | 0,056 |

(Val, WARM havuz, 438 sorgu; `reports/lexical_lever_probe_val.json`. ISCO sınıflandırıcısının val CV'lerinde ilk-1
doğruluğu %59,1. Tablo ile JSON ayrışırsa JSON geçerlidir.)

**Özet** (`lexical_lever_probe_val.json`): K=50'de ISCO-4 özniteliği NDCG@10'u +4,8 (60,49 → 65,27), Top-1'i +5,0
puan (p = 0,006) artırır. kNN daha çok artırır (NDCG +6,6; Top-1 +5,5, p = 0,010); ikisi birlikte kNN'e neredeyse bir
şey eklemez (67,14). Bu üç varyantta NDCG için p < 0,001. K=100 base NDCG'yi az artırır (+1,0; p = 0,038), Top-1'de
anlamlı değildir (p = 0,28). Mutlak düzey (≈65–69) hedefin (92) çok altındadır: leksikal hat tek başına yetmez; nöral
aşamalar şarttır.

**Okuma notları:**
- Bu hat nöral değildir. Dense ve CE öznitelikleri eklenince kazançlar küçülebilir ya da başka bir yere kayabilir.
  Kaldıracı A100 hattına taşıdıktan sonra val'de yeniden doğrula (§10.1 `VAL_COMP`, `per_query_val.csv`).
- **ISCO-4 özniteliği** (`scripts/13`): train CV'lerinin `cv_struct` metninde TF-IDF (kelime 1-2 gram, `min_df=2`,
  `max_features=100000`, `sublinear_tf`, `preprocessor=core.tr_ascii`) + `LinearSVC(C=1.0)` ile `gold_isco4` öğrenilir.
  Olasılıklar karar değerlerinin sıcaklıklı softmax'ıdır (`SoftSVC`, T = 0,1; LinearSVC'nin `predict_proba`'sı
  yoktur). Sınıflandırıcı CV'ye `cv_struct`, ilana `job_struct` üzerinden uygulanır. Öznitelikler: `isco_dot` (iki
  olasılık vektörünün iç çarpımı) ve `isco_top1_match` (ilk-1 sınıf eşleşmesi). İlan tarafının altın etiketi
  kullanılmaz. Bu yol `AttributeModels`'tan farklıdır: orada sektör sınıflandırıcısı (LogisticRegression) yalnız CV'ye
  uygulanır, ilanda ilanın kendi `industry` alanı kullanılır (`JobTable.build`); yaka ise unvanlarda karakter n-gram
  LogisticRegression ile iki tarafa uygulanır. K2'yi çekirdeğe taşırken ölçülen tarifi (LinearSVC + T = 0,1) koru;
  sınıflandırıcıyı değiştirirsen val'de yeniden ölç.
- **kNN etiket aktarımı:** `cv_struct` TF-IDF kosinüsüyle en yakın 30 train CV'si bulunur; ağırlık benzerlik²'dir
  (toplamı 1'e normalize edilir). Komşuların o ilana verdiği derecelerden (g) üç öznitelik çıkar: `knn_gain` =
  (2^g − 1)/7, `knn_rel` = [g ≥ 2], `knn_g3` = [g = 3] değerlerinin ağırlıklı ortalaması. Cold ilan sütunları 0'dır.
  Cold, yani yeni ilanlara hiç sinyal vermez. Ürün davranışını değiştirdiği için kullanmadan önce kullanıcıya sor (`00` §4). Ayrıca
  çıkarım paketine train dereceleri (2.835 × 3.000) eklenmesi gerekir.

## 3. Kaldıraçlar (kanıt düzeyi ve maliyete göre)

Kanıt düzeyleri: **V** = bu repoda val'de ölçüldü (leksikal hat, WARM havuz, `scripts/13`); **L** = literatürde güçlü
kanıt, bu veride ölçülmedi; **?** = bilinmiyor.

| # | Kaldıraç | Nerede | Mekanizma | Kanıt | Maliyet | Risk |
|---|---|---|---|---|---|---|
| K1 | `N_CANDIDATES` 50 → 100 | `CFG` (pakete `EXPORT_CFG` → `n_candidates` ile geçer) | Aday tavanını yükseltir (val WARM BM25 kâhin: 84,14 → 90,67) | V + tavan | CE değerlendirme/çıkarım ×2 | Düşük; çıkarımda top_k sınırı da değişir |
| K2 | ISCO-4 eşleşme özniteliği | `core.AttributeModels` + `FEATURE_COLUMNS` + `build_pair_features` | grade ≥ 2 ≈ aynı ISCO-4 | V | CPU'da saniyeler; çekirdek ve paket değişir | Düşük (yalnız train etiketleri) |
| K3 | Uzmanlık (`occ_key`) düzeyinde öznitelik: unvan/uzmanlık sınıflandırıcısı ya da en yakın train unvanı | aynı | grade 3 = aynı `occ_key` (+ lokasyon) | ? | Saniyeler–dakikalar | 1.349 sınıf (train `gold_occ_key`, sentetik dahil; `clean_cvs.parquet`, rapor dosyasında yok), seyrek; görülmemiş uzmanlıklar |
| K4 | kNN etiket aktarımı (öznitelik ya da 3. RRF kanalı) | çekirdek + paket | Benzer CV'lerin dereceleri | V | Saniyeler; paket büyür | **Kullanıcı kararı** (cold ilanlara karşı yanlılık) |
| K5 | RRF ağırlık / `k` ayarı | `CFG` + `hybrid()` + `EXPORT_CFG` + `SOTAMatcher.match` (`core.rrf_scores(weights=…)`) | Güçlü kanala daha çok ağırlık | ? | Saniyeler | Düşük; ağırlık `EXPORT_CFG`'ye ve `SOTAMatcher.match`'e girmezse paket notebook'tan farklı sıralar (çekirdek değişikliği). `k` zaten `rrf_k` olarak aktarılır |
| K6 | Aşama 1: daha fazla pozitif (QRELS'ten), qrels ile tam yanlış-negatif maskesi, ikinci tur zor negatif (ANCE/RocketQA), CE → bi damıtma (Margin-MSE) | `build_mnrl_rows`, kayıp | Daha çok ve daha temiz sinyal, daha iyi geri getirme | L | 1–2× Aşama 1 | Grade-2 pozitifleri gürültülü |
| K7 | CE eğitim listelerini dağılım uyumlu kur: base/BM25 adayları ya da çapraz uydurulmuş BI | §9 hücresi | Train listeleri val/test kadar zor olur (`05` B.6) | ? | 0 ya da K× Aşama 1 | Düşük |
| K8 | `CE_DOCS_PER_QUERY` 16 → 32; `LambdaLoss(k=10)`; `CE_EPOCHS` 2 → 3; `CE_LR` taraması | `CFG`, §9 | Daha çok negatif, ilk-10'a odak | ? | ≈2× CE | Bellek (`02` §4) |
| K9 | `CE_MAX_LEN` 768–1024 / akıllı sıkıştırma (Pozisyon/Kıdem/Konum/Yetkinlikler önde) | `CFG`, `core.compact_*_for_ce` | Kırpılma kaybını azaltır (önce §9.1'de ölç) | ? | 2–4× CE | Bellek |
| K10 | Daha büyük reranker (bge-reranker-v2-gemma, Qwen3-Reranker 4B/8B) — sıfır-atış skoru LTR özniteliği olarak ya da LoRA ile ince ayar | §9, çekirdek | Daha güçlü çapraz dikkat | L (genel kıyaslamalar) | Saatler; 40 GB'ta LoRA gerekebilir | **Lisans kontrolü**; peft sabit değil |
| K11 | LTR'yi çapraz uydurulmuş train sorgularında eğit (kat dışı dense/CE öznitelikleri) | §10 + Aşama 1/2a K-katlı | val'in 457 sorgusundan çok daha fazla LTR verisi | ? (val'de ölçülmedi) | ≈5× Aşama 1+2a | Kat dışı olmayan train öznitelikleri **yasak** (sızıntı) |
| K12 | Daha güçlü embedder (Qwen3-Embedding, multilingual-e5-large-instruct, Türkçe modeller) | `BI_ENCODER_BASE` + `prefixes()` | Daha iyi geri getirme | L | Tam çalıştırma | Önek/talimat biçimi; lisans |
| K13 | BGE-M3 seyrek / çok-vektör modları | FlagEmbedding (sabit değil) | Leksikal + çok-vektör sinyali | L (M3 makalesi) | Orta | Yeni bağımlılık |
| K14 | Topluluk / test-zamanı çoğaltma (başlık-yalnız, beceri-yalnız sorgu görünümleri) | §8/§10 | Varyansı düşürür | ? | Üye sayısıyla doğrusal | Düşük |
| K15 | Dış LLM ile liste-bazlı yeniden sıralama (RankGPT) | yeni | Güçlü sıralayıcı | L | API maliyeti | **Kullanıcı kararı** (gizlilik). Etiketler de LLM kökenli olduğundan dairesellik riski var; ayrı bir sistem olarak raporla |

- **K11:** leksikal hatta train çapraz uydurma yalnız devir hazırlığındaki test keşfinde denendi (`06` E000); o sonuç
  kanıt sayılmaz. Önce yalnız-val bir ölçümle doğrula (ör. `scripts/13`'e bir varyant).
- **Çekirdeğe dokunan kaldıraçlar** (K2–K5, K9, K10): çekirdek veri üreticisinin de parçasıdır (`05` onu içe aktarır:
  tam liste `CLAUDE.md` kural 5). `05`'ten önce `data_bundle/metadata.json`'u yedekle, sonra `dosyalar` altındaki
  parquet sha256'larını karşılaştır; değiştiyse bu bir veri değişikliğidir (`00` §7). Colab'da §2'deki
  `paket parmak izi` repodakiyle aynı olmalı.
- **Maliyetler tahmindir;** GPU'da ölçülmedi. İlk çalıştırmadan sonra `run_summary.json` → `hucre_sureleri` hücre
  başına süreyi ve GPU tepesini verir (`sureler_sn` yalnız τ, Bi-encoder ve Cross-encoder eğitimlerini tutar).
- Yeni sentetik veri, dış LLM, kısıtlayıcı lisans ve ürün davranışını değiştiren yanlılık kullanıcı kararıdır
  (`00` §4).

Model adlarını ve lisanslarını kullanmadan önce model kartında doğrula. Hazırlık sırasında huggingface.co'ya
erişilemedi; parametre sayıları ve lisanslar ikincil kaynaklardan alındı.

## 4. Kestirme yollar: yapma

Bunlar sayıyı yükseltir ama modeli iyileştirmez. İki türü vardır.

**Tanım değişiklikleri** (kullanıcının kararı, `00` §4). Gerekçeli bir yeniden tanım ancak kullanıcı
onayıyla ve eski ve yeni sayılar ayrı raporlanarak yapılabilir:
- **`REL_THRESHOLD=1`.** Kâhinin P@10'u 83,38'den 99,41'e çıkar (test; `metric_ceilings.json` → `kahin`, `esik2` →
  `esik1`). Model hiç değişmez; tanım değişir.
- **Alakalısı < 10 olan sorguları atmak** (P@10 kâhini 100 olur) ya da başka bir alt-küme seçmek.
- **`grade_matrix()` kurallarını hedefi tutturacak biçimde değiştirmek.** Örnek: lokasyonu grade 3'ten çıkarmak. Val'de
  aynı uzmanlık/yaka/kıdem çiftlerinin %71,3'ü yalnız lokasyon yüzünden grade 2'dedir (`grade3_lokasyon_bagimliligi`);
  bunlar grade 3 olur ve grade-3'ü olan sorguların payı (val %31,7; `faz2_veri_ozeti.json` →
  `grade3_ilani_olan_cv_orani`) kabaca 2,5 katına çıkar (dosyaya kayıtlı değil; `scripts/12` §5'teki eşleşme kuralıyla
  yeniden hesaplanabilir).

**Metodoloji ihlalleri** (kullanıcı onayıyla da yapılmaz):
- **İlan tarafının altın etiketlerini (`gold_*`, `legacy_*`) öznitelik yapmak.** Kuralın yarısını doğrudan
  gözlemlenebilir kılar. Kullanıcıya sorulacak bir seçenek değildir; kullanıcı ısrar ederse ayrı, açıkça etiketlenmiş
  bir sistem olur, ana sonuç olmaz.
- **Test sayısına bakıp yapılandırma, kaldıraç ya da hedefli veri seçmek** (§11, §11.1, §12, `per_query_test.csv`
  dahil) ya da birden çok çalıştırmadan testte en iyisini raporlamak.
- **Devir hazırlığındaki test keşif sayılarını (`06` E000) kanıt olarak kullanmak.**
- **Cold ilanları ya da test CV'lerini herhangi bir eğitim veya seçim sinyaline sokmak.**

## 5. Hedef yine tutmazsa: dürüst rapor

Bu bir başarısızlık değil, bir bulgudur, yeter ki ölçülmüş ve açıklanmış olsun. Raporda şunlar bulunmalı:

1. Elde edilen değerler (test, tek seferlik okuma; `run_summary.json` → `hedefler`) ve %95 GA'ları; test kümesiyle
   temasların dürüst sayımı (`06` E000, `test_degerlendirme_sayisi`).
2. Her hedefin tavanı:
   - kâhin;
   - ikili kusursuz;
   - aday tavanı (§9.1: `aday_tavani_val(WARM)` karar için, `aday_tavani_test(tam_havuz,yalniz_rapor)` rapor için);
   - tavana oran.
3. Bileşen katkıları ve p-değerleri: test `ablation_component_tests.csv` (rapor), val `val_component_tests.csv`
   (kararların dayanağı).
4. Denenen kaldıraçlar ve val etkileri (`06` günlüğünden tablo; çalıştırmalar arası p-değeri `per_query_val.csv`'den),
   toplam tam çalıştırma sayısı.
5. Hata analizi: hangi kovalar kaybettiriyor (kararlar val kovalarına dayanır; raporda test kovaları da gösterilebilir),
   bunların ne kadarı etiket gürültüsü (kör doğrulama: kural grade-3 çiftlerinin 47/90'ı yargıçça 2;
   `reports/qrels_validation.json`).
6. Kalan hipotezler ve maliyetleri.
7. Kullanıcıya **soru**, cevabı kullanıcı verir. Örnek:
   > "P@10 > 85 bu qrels tanımıyla imkânsız (kâhin 83,38). Hedefi P@10_capped > 85 ya da 'kâhine oranla
   > P@10 > %X' olarak yeniden ifade etmeyi öneriyorum. Onaylıyor musunuz?"
