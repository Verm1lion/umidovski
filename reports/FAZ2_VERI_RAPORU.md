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
| Veri setindeki CV yakası ↔ anotatör yakası | 0.9599 | 0.9823 |


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


qrels (grade ≥ 1): **608,901** satır; dağılım {'1': 472005, '2': 131784, '3': 5112}. Listede olmayan tüm çiftler grade 0'dır. **Tüm** CV × ilan çiftleri yargılıdır (tam yargı).

### 3.1 Kural-türevi derecelerin bağımsız çift-düzeyi yargılarla doğrulanması

Tabakalı örneklem (her dereceden eşit sayıda, yalnız gerçek val/test CV'leri), n = 360. Yargıçlar kural derecesini görmeden yalnızca CV ve ilan metnini okuyup 0–3 verdi (`annotation/qrels_validation/RUBRIC.txt`).


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


**Yorum:** Uyum yüksektir (ağırlıklı κ ve |fark| ≤ 1 oranı). Ayrışmaların neredeyse tamamı **bir kademe aşağı** yöndedir: bağımsız yargıçlar kural grade-3 çiftlerinin bir kısmını 2, kural grade-2 çiftlerinin bir kısmını 1 olarak değerlendirmiştir (ör. aynı uzmanlık ama farklı teknoloji yığını; aynı ISCO grubu ama farklı alt rol). Kurallar bu anlamda biraz cömerttir; bu, mutlak NDCG değerlerini etkiler ancak sistemlerin **göreli** sıralamasını değiştirmez, çünkü tüm sistemler aynı qrels ile ölçülür. Ters yönde hata çok azdır: kural grade-0 çiftlerin 87/90'i yargıçça da 0, hiçbiri ≥ 2 değildir; yargıcın alakalı (≥ 2) bulup kuralın ≤ 1 dediği çift sayısı yalnızca 7'dir.

## 4. Sızıntısız bölme

| Bölüm | CV | gerçek | sentetik | ort. grade-3 ilan | ort. grade-2 ilan | grade-3'ü olan CV oranı | alakalısı olmayan CV | medyan alakalı (≥2) |
|---|---|---|---|---|---|---|---|---|
| train | 2838 | 2127 | 711 | 1.41 | 34.32 | 0.5106 | 49 | 27.0 |
| val | 455 | 455 | 0 | 1.19 | 37.91 | 0.3363 | 10 | 27.0 |
| test | 461 | 461 | 0 | 1.26 | 37.17 | 0.3774 | 8 | 27.0 |


İlan havuzu: {'warm': 2699, 'cold': 301} (cold = hiçbir eğitim sinyalinde kullanılmaz). Vitrin adayları (test'e sabit): [20951, 10587, 10511, 21029].

### 4.1 Otomatik sızıntı kontrolleri (hepsi `05_build_dataset.py` içinde zorunlu; biri bile başarısızsa paket üretilmez)

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
| test_train_max_cos_medyan | 0.247 |
| test_train_max_cos_p90 | 0.6 |
| test_cv_cos_ge_0.8_trainde_ikizi | 0 |
| sentetik_test_max_cos_p99 | 0.19 |
| sentetik_test_cos_ge_0.8 | 0 |
| TUM_KONTROLLER_GECTI | True |

## 5. Zor negatifler ve triplet'ler

* Train triplet: **4,886**, val triplet (yalnız val kaybı için): 444. İstatistik: {'trap_fallback_bm25': 23, 'anchor_pozitifsiz_atlandi': 53, 'anchor': 2838}.

* `neg_lexical`: Ayni/benzer unvan sozcugu (karakter n-gram), FARKLI sektor, grade 0 (Baklava Ustasi ~ Siva Ustasi)

* `neg_seniority_collar`: Ayni sektor veya ayni ISCO alt-ana grubu; yaka zit veya |kidem farki|>=3, grade<=1

* `neg_bm25`: BM25 ile en yuksek skorlu fakat grade 0 ilanlar (kelime kesisimi yuksek, anlamca uyumsuz)

* Negatiflerin derece dağılımı: {'neg_lexical_grade': {'0': 4886}, 'neg_seniority_collar_grade': {'0': 3206, '1': 1680}, 'neg_bm25_grade': {'0': 4886}}

* Notebook ayrıca 4. bir negatif üretir: eğitimsiz base modelin üst sıralara koyduğu **grade-0** ilanlar (qrels ile filtrelendiği için yanlış negatif yok).

## 6. Sentetik CV zenginleştirmesi (yalnız train)

* Hedef: train'de hiç / tek CV'si olmayan ve WARM ilanlarda geçen uzmanlıklar. Üretilen: 868, kabul edilen: **711**.

* Gidiş-dönüş (kör anotatör) tutarlılığı: ISCO 727, yaka 845, kıdem(±1) 863; filtre reddi 157, val/test başlık çakışması reddi 0.

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
| NDCG@10 | 51.63 |
| MRR@10 | 71.15 |
| P@10 | 38.21 |
| P@10_capped | 46.04 |
| Top1 | 60.26 |
| Top1_exact(grade3) | 9.93 |
| R@10 | 22.85 |
| R@50 | 44.94 |
| R@100 | 56.83 |
| P@10(sektor) | 63.77 |
| Top1(sektor) | 77.01 |
| n_queries | 453 |
| n_skipped_no_relevant | 8 |
| NDCG@10_CI95 | [49.2, 54.0] |


Aynı BM25 sıralaması eski metrikle ("aynı sektör") ölçülünce çok daha yüksek görünür. Yeni kademeli qrels meslek, yaka, kıdem ve lokasyonu ölçtüğü için çok daha katıdır. Nöral sistemlerin sayıları Colab A100 çalıştırmasıyla notebook'ta üretilir.

### 8.1 Nöral model olmadan öğrenilen sıralamanın katkısı: BM25 + LambdaMART (yerel, gerçek ölçüm)

`scripts/11_local_lexical_ltr.py`: BM25 Top-50 adaylar, metin-türevli öznitelikler (dense/CE = 0), LambdaMART val sorgularında (cold hariç) eğitildi, test bir kez ölçüldü.


| Metrik | BM25 | BM25 + LambdaMART (leksikal, noral yok) |
|---|---|---|
| NDCG@10 | 51.63 | 64.58 |
| MRR@10 | 71.15 | 81.16 |
| P@10 | 38.21 | 45.47 |
| P@10_capped | 46.04 | 54.71 |
| R@10 | 22.85 | 27.0 |
| Top1 | 60.26 | 74.39 |
| Top1_exact(grade3) | 9.93 | 17.66 |
| NDCG@10_CI95 | [49.2, 54.0] | [62.2, 66.8] |


NDCG@10 farkı için Fisher randomizasyon testi p = 9.999e-05. En önemli öznitelikler (gain): ind_prob_job_sector, collar_agree, title_sim, bm25_z, title_tok_overlap, job_desc_len_log. Sektör/yaka öznitelikleri yalnızca train CV'lerinde eğitilmiş sınıflandırıcıların **tahminleridir**.

## 9. Paket içeriği

| Dosya | Boyut | SHA-256 |
|---|---|---|
| clean_cvs.parquet | 4.64 MB | 45c1a3a195b2b0c6… |
| clean_jobs.parquet | 7.45 MB | c6c7d073c802abf5… |
| legacy_train_pairs.parquet | 0.07 MB | b75994ad7d839af9… |
| match_candidate_sota.py | 0.05 MB | 02b865f8240dd04d… |
| qrels_ground_truth.parquet | 0.36 MB | cb94e071de82dc1a… |
| test_queries.parquet | 0.26 MB | 7ab23c8a817c806b… |
| train_queries.parquet | 1.84 MB | 46c884beaf619e67… |
| train_triplets.parquet | 0.08 MB | 8cfbf451c7207df7… |
| val_queries.parquet | 0.26 MB | d7366c1f2c6a2e25… |
| val_triplets.parquet | 0.02 MB | 3bc41bf704eaace2… |
