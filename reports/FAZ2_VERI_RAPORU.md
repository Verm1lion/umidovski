# FAZ 2 — Sızıntısız Veri Mühendisliği, Kademeli Ground Truth ve Zor Negatifler

> Bu rapor `scripts/10_make_faz2_report.py` tarafından üretilmiş JSON özetlerinden otomatik yazılır. Ham sayılar: `data_bundle/metadata.json`, `reports/annotation_*.json`, `reports/synthetic_cv_stats.json`, `reports/qrels_validation.json`, `reports/local_bm25_test.json`.

## 1. Boru hattı

```
01 SQL ayrıştırma ─► 02 anotasyon öğeleri ─► [ISCO-08 anotasyonu, 68 parça] ─► 04 birleştirme + IAA
   ─► [konsolidasyon, 43 ISCO grubu] ─► 04 finalize ─► 06 sentetik CV hedefleri ─► [üretim + kör doğrulama]
   ─► 06 merge (gidiş-dönüş filtresi) ─► 05 veri paketi (split, qrels, triplet, sızıntı kontrolleri) ─► data_bundle.zip
```

## 2. Meslek anotasyonu (ISCO-08 + uzmanlık + yaka + kıdem)

* Kılavuz: `annotation/GUIDELINES.md`. 6.043 öğe (3.043 CV + 3.000 ilan), aynı sektör ve benzer unvanlar aynı parçada olacak şekilde 68 parça. Ham kodlar `annotation/raw/`.

* Konsolidasyon sonrası: **298 ISCO-08 birim grubu**, **1630 kanonik uzmanlık**; hem CV hem ilan tarafında bulunan uzmanlık: 877, ISCO grubu: 236. Konsolidasyonda ISCO kodu değişen öğe: 171, uzmanlık adı değişen: 1447.

### 2.1 Anotatörler arası uyum (rastgele %10 alt küme, bağımsız ikinci anotasyon, farklı parça bileşimi)

n = 604 öğe.

| Alan | Cohen κ | Birebir uyum |
|---|---|---|
| ISCO-08 4 hane (birim grup) | 0.9159 | 0.9172 |
| ISCO-08 3 hane | 0.9296 | — |
| ISCO-08 2 hane | 0.9398 | — |
| Yaka (BLUE/WHITE) | 0.9799 | 0.9917 |
| Kıdem 0–5 (ikinci dereceden ağırlıklı κ) | 0.9585 | 0.9205 |


κ değerleri ikinci anotasyon ile ana anotasyonun **konsolidasyon öncesi** kodları arasındadır. Ayrıca veri setindeki CV `collar` alanı anotatör yakasıyla tüm 3.043 CV üzerinde κ = 0.9599 (uyum 0.9823) uyumludur; bu anotatörler arası değil, bir geçerlilik kontrolüdür.


Kıdemde |fark| ≤ 1 oranı: 0.9983. Landis & Koch (1977) ölçeğinde κ > 0,80 'neredeyse mükemmel' uyumdur. Uzmanlık adlarının konsolidasyon öncesi birebir yazım uyumu 0.6192'dir; bu yüzden ad farklılıkları ayrı bir konsolidasyon turunda kanonikleştirilmiştir (ISCO grubu düzeyinde).


**Yaka (gold):** Yaka mesleğin bir özelliğidir; aynı uzmanlıktaki tüm öğeler için çoğunluk oyu (CV + ilan birlikte) kullanılır. Veri setindeki `collar` alanı ile uyum yukarıdadır.


**Anotatörlerin veri kalitesi notları:** Birçok sentetik CV'de deneyim/eğitim metni unvanla uyuşmayan şablon gürültüsü taşıyor (ör. astrofizik unvanlı CV'de bulut/veri deneyimi). Bu CV'ler unvan + özet + becerilerden kodlandı ve `conf` düşürüldü. Güven dağılımı: {'3': 2991, '2': 2752, '1': 300}.

## 3. Kademeli alaka (graded relevance) kuralları

| Grade | Kural |
|---|---|
| 3 | Tam eslesme: ayni uzmanlik (occ_key) + yaka uyumu + \|kidem farki\| <= 1 + lokasyon uyumu (ayni sehir veya Remote) |
| 2 | Yuksek alaka: ayni ISCO-08 birim grubu (isco4) + yaka uyumu + \|kidem farki\| <= 2 (kidem/lokasyonda tolere edilebilir fark) |
| 1 | Kismi alaka: yaka uyumu + (ayni ana sektor VEYA ayni isco4 fakat kidem farki >= 3) |
| 0 | Alakasiz / uygunsuz: diger tum durumlar (farkli yaka, farkli sektor ve farkli meslek) |


qrels (grade ≥ 1): **608,691** satır; dağılım {'1': 471829, '2': 131751, '3': 5111}. Listede olmayan tüm çiftler grade 0'dır. **Tüm** CV × ilan çiftleri yargılıdır (tam yargı).

### 3.1 Kural-türevi derecelerin bağımsız çift-düzeyi yargılarla doğrulanması

Tabakalı örneklem (her dereceden eşit sayıda, yalnız gerçek — sentetik olmayan — CV'ler; yargılar hiçbir eğitimde kullanılmaz), n = 360. Yargıçlar kural derecesini görmeden yalnızca CV ve ilan metnini okuyup 0–3 verdi (`annotation/qrels_validation/RUBRIC.txt`).


| Ölçüt | Değer |
|---|---|
| İkinci dereceden ağırlıklı κ | 0.8357 |
| Doğrusal ağırlıklı κ | 0.7182 |
| Birebir uyum | 0.6806 |
| \|fark\| ≤ 1 | 0.9778 |
| İkili (grade ≥ 2) uyum / κ | 0.8611 / 0.7222 |
| Spearman ρ | 0.8753 |


Karışıklık matrisi (satır = kural, sütun = bağımsız yargı):

| kural \ yargı | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 | 87 | 3 | 0 | 0 |
| 1 | 15 | 68 | 7 | 0 |
| 2 | 8 | 35 | 47 | 0 |
| 3 | 0 | 0 | 47 | 43 |


**Yorum:** Ağırlıklı uyum yüksektir (κ, |fark| ≤ 1); birebir uyum ise %68.1 düzeyindedir. 115 ayrışmadan 97 tanesi (%84) bir kademe aşağı, 8 tanesi iki+ kademe aşağı, 10 tanesi yukarı yöndedir. Kural grade-3 çiftlerinden 47 / 90 tanesi yargıçça 2 bulunmuştur (ör. aynı uzmanlık ama farklı teknoloji yığını ya da kıdem/lokasyon yorumu); kural grade-2 çiftlerinden 43 / 90 tanesi 0–1 bulunmuştur. Yani kurallar özellikle üst derecelerde **cömerttir**: mutlak NDCG değerleri bir miktar iyimser olabilir. Tüm sistemler aynı qrels ile ölçüldüğü için karşılaştırma aynı ölçüt üzerindedir; ancak bu yanlılığın sistemleri eşit etkilediği kanıtlanmış değildir. Ters yönde: kural grade-0 çiftlerden 87 / 90 tanesi yargıçça da 0, hiçbiri ≥ 2 değildir. Not: grade-0 çiftler rastgele örneklendi; zor (BM25/leksikal) grade-0 çiftlerdeki hata oranını bu örneklem ölçmez. Yargıçlara derecelendirme rubriği verildiği için bu doğrulama kuralların tutarlı uygulanıp uygulanmadığını sınar; kuralların kendisinin doğruluğunu bağımsız olarak kanıtlamaz.

## 4. Sızıntısız bölme

| Bölüm | CV | gerçek | sentetik | ort. grade-3 ilan | ort. grade-2 ilan | grade-3'ü olan CV oranı | alakalısı olmayan CV | medyan alakalı (≥2) |
|---|---|---|---|---|---|---|---|---|
| train | 2835 | 2125 | 710 | 1.43 | 35.46 | 0.5203 | 43 | 27.0 |
| val | 457 | 457 | 0 | 1.07 | 32.3 | 0.3173 | 16 | 23.0 |
| test | 461 | 461 | 0 | 1.24 | 35.72 | 0.3362 | 8 | 27.0 |


İlan havuzu: {'warm': 2699, 'cold': 301} (cold = hiçbir eğitim sinyalinde kullanılmaz). Vitrin adayları (test'e sabit): [20951, 10587, 10511, 21029].

### 4.1 Otomatik sızıntı kontrolleri (sayım kontrolleri `05_build_dataset.py` içinde zorunludur ve 0 olmalıdır; biri bile başarısızsa paket üretilmez. `*_medyan`, `*_p90`, `*_p99` satırları yalnız bilgi amaçlıdır)

| Kontrol | Değer |
|---|---|
| id_kesisim_train_val | 0 |
| id_kesisim_train_test | 0 |
| id_kesisim_val_test | 0 |
| unvan_kesisim_train_test | 0 |
| unvan_kesisim_gercek_train_test | 0 |
| unvan_kesisim_train_val | 0 |
| sentetik_val_test_te | 0 |
| triplet_anchor_train_disi | 0 |
| val_triplet_anchor_val_disi | 0 |
| triplette_cold_ilan | 0 |
| legacy_pairs_train_disi_cv | 0 |
| legacy_pairs_cold_ilan | 0 |
| test_train_max_cos_medyan | 0.252 |
| test_train_max_cos_p90 | 0.605 |
| test_cv_cos_ge_0.8_trainde_ikizi | 0 |
| sentetik_test_max_cos_p99 | 0.169 |
| sentetik_test_cos_ge_0.8 | 0 |
| TUM_KONTROLLER_GECTI | True |

## 5. Zor negatifler ve triplet'ler

* Train triplet: **4,883**, val triplet (yalnız val kaybı için): 438. İstatistik: {'lex_fallback_bm25': 4, 'trap_fallback_bm25': 19, 'trap_fallback_bm25_satir': 29, 'anchor_pozitifsiz_atlandi': 49, 'en_cok_kullanilan_lex_negatif_satir': 25, 'en_cok_kullanilan_trap_negatif_satir': 114, 'en_cok_kullanilan_bm25_negatif_satir': 25, 'anchor': 2835}.

* `neg_lexical`: Ayni/benzer unvan sozcugu (karakter n-gram TF-IDF kosinus >= 0.05), FARKLI sektor, grade 0 (Baklava Ustasi ~ Siva Ustasi); benzerlik neg_lexical_title_sim sutununda

* `neg_seniority_collar`: Ayni sektor veya ayni ISCO alt-ana grubu; yaka zit veya |kidem farki|>=3, grade<=1. Boyle ilan yoksa BM25 sirasi 4-10'dan grade 0 ilan konur (neg_seniority_collar_is_fallback=True)

* `neg_bm25`: BM25 ile en yuksek skorlu fakat grade 0 ilanlar (kelime kesisimi yuksek, anlamca uyumsuz)

* Negatiflerin derece dağılımı: {'neg_lexical_grade': {'0': 4883}, 'neg_seniority_collar_grade': {'0': 3153, '1': 1730}, 'neg_bm25_grade': {'0': 4883}}

* Notebook ayrıca 4. bir negatif üretir: eğitimsiz base modelin üst sıralara koyduğu **grade-0** ilanlar (qrels ile filtrelenir: kurallara göre grade 0'dır; yine de en benzer grade-0 ilanların bir kısmı yakın-alakalı olabilir).

## 6. Sentetik CV zenginleştirmesi (yalnız train)

* Hedef: train'de hiç / tek CV'si olmayan ve WARM ilanlarda geçen uzmanlıklar. Üretilen: 868, kabul edilen: **710**.

* Gidiş-dönüş (kör anotatör) tutarlılığı: ISCO 727, yaka 845, kıdem(±1) 863; filtre reddi 157, val/test başlık çakışması reddi 1.

## 7. Vitrin adayları için kural dereceleri (kontrol)

**20951 — Gaziantep Baklava & Yufka Ustası** (split=test, uzmanlık: Baklava Ustası, ISCO 7512, kıdem 3; grade-3 ilan: 0, grade-2: 16)

* Alakalı örnekler: Taş Fırın Pide ve Lahmacun Ustası -> 2; Kol Böreği ve Boşnak Böreği Ustası -> 2; Geleneksel Şerbetli Tatlı Ustası -> 2; Taş Fırın ve Ekmek Pişirme Ustası -> 2; Gaziantep Usulü Baklava Ustası -> 2; Pide ve Lahmacun Ustası -> 2

* Tuzak ilanlar ve dereceleri: Tuğla ve Duvar Ustası → 0; CNC Torna ve Dik İşleme Operatörü → 0; CNC Abkant Büküm Ustası → 0; Distribütör ve Bayi Yöneticisi → 0; Medikal Direktör (İlaç Sektörü) → 0; İmalat ve Talaşlı Üretim Mühendisi (CNC) → 0; Kurumsal Risk Yönetimi (ERM) Direktörü → 0; CNC Lazer Kesim Operatörü → 0


**10587 — Flight Operations Dispatcher** (split=test, uzmanlık: Uçuş Harekat Uzmanı (Dispeçer), ISCO 4323, kıdem 1; grade-3 ilan: 1, grade-2: 11)

* Alakalı örnekler: Uçuş Harekat Uzmanı (Flight Dispatcher) -> 3; Sevkiyat ve Lojistik Planlama Uzmanı -> 2; Konteyner Gemisi Yük Planlama ve Lashing Sorumlusu -> 2; Hava Kargo Operasyon ve Lojistik Görevlisi -> 2; Hava Trafik Kontrolörü / Uçuş Harekat Uzmanı (Dispatcher) -> 2; Uçuş Ekibi Planlama ve Otel Tahsis Uzmanı -> 2

* Tuzak ilanlar ve dereceleri: Tuğla ve Duvar Ustası → 0; CNC Torna ve Dik İşleme Operatörü → 0; CNC Abkant Büküm Ustası → 0; Distribütör ve Bayi Yöneticisi → 0; Medikal Direktör (İlaç Sektörü) → 0; İmalat ve Talaşlı Üretim Mühendisi (CNC) → 0; Kurumsal Risk Yönetimi (ERM) Direktörü → 0; CNC Lazer Kesim Operatörü → 0


**10511 — Java Spring Boot Geliştirici** (split=test, uzmanlık: Backend Yazılım Geliştirici, ISCO 2512, kıdem 2; grade-3 ilan: 23, grade-2: 182)

* Alakalı örnekler: Java Yazılım Uzmanı (Spring Boot / Kafka) -> 3; Kıdemli Java Backend Mühendisi (Dağıtık Sistemler & Kubernetes) -> 3; Backend Yazılım Mühendisi (Java 21 / Quarkus) -> 3; Kıdemli Java Yazılım Uzmanı (Büyük Veri & Event Streaming) -> 3; .NET Yazılım Uzmanı (.NET 8 / Web API / RabbitMQ) -> 3; Backend Geliştirici (.NET 8 / gRPC & Event-Driven) -> 3

* Tuzak ilanlar ve dereceleri: Tuğla ve Duvar Ustası → 0; CNC Torna ve Dik İşleme Operatörü → 0; CNC Abkant Büküm Ustası → 0; Distribütör ve Bayi Yöneticisi → 0; Medikal Direktör (İlaç Sektörü) → 0; İmalat ve Talaşlı Üretim Mühendisi (CNC) → 0; Kurumsal Risk Yönetimi (ERM) Direktörü → 0; CNC Lazer Kesim Operatörü → 0


**21029 — Java ve Spring Boot Yazılım Stajyeri** (split=test, uzmanlık: Backend Yazılım Geliştirici, ISCO 2512, kıdem 0; grade-3 ilan: 15, grade-2: 110)

* Alakalı örnekler: Junior Java Geliştirici (Spring Boot / Mikroservis) -> 3; Java Yazılım Stajyeri (Spring Boot & Cloud - Yaz Dönemi) -> 3; Junior .NET Core Geliştirici (C# / Entity Framework) -> 3; .NET Yazılım Stajyeri (Ar-Ge Merkezi - Yaz & Kış Dönemi) -> 3; Junior Backend Geliştirici (C# / MSSQL - Kurumsal ERP) -> 3; Junior .NET Geliştirici (Bulut Tabanlı Muhasebe API'ları) -> 3

* Tuzak ilanlar ve dereceleri: Tuğla ve Duvar Ustası → 0; CNC Torna ve Dik İşleme Operatörü → 0; CNC Abkant Büküm Ustası → 0; Distribütör ve Bayi Yöneticisi → 0; Medikal Direktör (İlaç Sektörü) → 0; İmalat ve Talaşlı Üretim Mühendisi (CNC) → 0; Kurumsal Risk Yönetimi (ERM) Direktörü → 0; CNC Lazer Kesim Operatörü → 0


## 8. Yeni test kümesinde yerel referans ölçüm: BM25 (GPU gerekmez, gerçek sayı)

| Metrik | BM25 (test) |
|---|---|
| NDCG@10 | 53.66 |
| MRR@10 | 74.03 |
| P@10 | 38.37 |
| P@10_capped | 48.53 |
| Top1 | 64.68 |
| Top1_exact(grade3) | 9.49 |
| R@10 | 26.59 |
| R@50 | 47.46 |
| R@100 | 59.05 |
| P@10(sektor) | 64.12 |
| Top1(sektor) | 78.09 |
| n_queries | 453 |
| n_skipped_no_relevant | 8 |
| NDCG@10_CI95 | [51.5, 55.8] |


Aynı BM25 sıralaması eski metrikle ("aynı sektör") ölçülünce çok daha yüksek görünür. Yeni kademeli qrels meslek, yaka, kıdem ve lokasyonu ölçtüğü için çok daha katıdır. Nöral sistemlerin sayıları Colab A100 çalıştırmasıyla notebook'ta üretilir.

### 8.1 Nöral model olmadan öğrenilen sıralamanın katkısı: BM25 + LambdaMART (yerel, gerçek ölçüm)

`scripts/11_local_lexical_ltr.py`: BM25 Top-50 adaylar, metin-türevli öznitelikler (dense/CE = 0), LambdaMART val sorgularında (cold hariç) eğitildi, test bir kez ölçüldü.


| Metrik | BM25 | BM25 + LambdaMART (leksikal, noral yok) |
|---|---|---|
| NDCG@10 | 53.66 | 66.05 |
| MRR@10 | 74.03 | 80.23 |
| P@10 | 38.37 | 45.01 |
| P@10_capped | 48.53 | 56.35 |
| R@10 | 26.59 | 29.71 |
| Top1 | 64.68 | 72.85 |
| Top1_exact(grade3) | 9.49 | 15.67 |
| NDCG@10_CI95 | [51.5, 55.8] | [63.9, 68.2] |


NDCG@10 farkı için Fisher randomizasyon testi p = 9.999e-05. En önemli öznitelikler (gain): ind_prob_job_sector, bm25_z, title_sim, collar_agree, title_tok_overlap, job_p_blue. Sektör/yaka öznitelikleri yalnızca train CV'lerinde eğitilmiş sınıflandırıcıların **tahminleridir**.

## 9. Paket içeriği

| Dosya | Boyut | SHA-256 |
|---|---|---|
| clean_cvs.parquet | 4.64 MB | 88b004e5b5ebe329… |
| clean_jobs.parquet | 7.45 MB | c6c7d073c802abf5… |
| legacy_train_pairs.parquet | 0.07 MB | 4a30d5faf3552d5d… |
| match_candidate_sota.py | 0.05 MB | 4b4621598befa131… |
| qrels_ground_truth.parquet | 0.36 MB | 108637bac126ee38… |
| test_queries.parquet | 0.26 MB | 6e41d74a7d9c4062… |
| train_queries.parquet | 1.83 MB | 807bf528da34164e… |
| train_triplets.parquet | 0.10 MB | f3f18b14bb7ad033… |
| val_queries.parquet | 0.26 MB | 80dbce89f6adca9c… |
| val_triplets.parquet | 0.02 MB | 85519f8f0bb88dc9… |
