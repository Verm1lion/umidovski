# FAZ 1 — Mevcut Sistemin Adli Denetimi (Forensic Audit)

> Ölçüm sayıları, aşağıda sayılan istisnalar dışında, `scripts/03_forensic_audit.py` ile yeniden üretilebilir; ham çıktı `reports/faz1_audit.json`.
> Birkaç statik gözlem doğrudan kaynak dosyalardan okunmuştur: eski rehberin bildirdiği metrikler ve çarpanlar
> (`MODEL_VE_TEST_REHBERI.md`), eski bölme özetindeki train grup sayısı ve grup büyüklükleri (`veriseti/train_groups_summary.json → group_count`, `groups[].count`) ve
> "Distribütör ve Bayi Yöneticisi" ilanının açıklama uzunluğu (`work/tum_ilanlar.parquet`). Betik çıktısında yer
> almayan, doğrudan veri ve kod dosyaları üzerinde yapılmış birkaç ek kontrol metinde *(ayrıca kontrol edildi)*
> diye işaretlenmiştir.
> Eski değerlendirme fonksiyonları (`test.py`) **değiştirilmeden içe aktarılarak** kullanılmıştır. Ortamda eski
> modelin ağırlıkları olmadığından, kanıtlar **modelden bağımsız** kurgulanmıştır: sıralayıcı olarak rastgele skor
> ve BM25 kullanılmıştır. Gösterilen hatalar modelin değil, **metriklerin ve veri kurgusunun** hatalarıdır.

```bash
python scripts/01_parse_jobs_sql.py      # SQL -> work/tum_ilanlar.parquet / .csv
python scripts/03_forensic_audit.py      # -> reports/faz1_audit.json + reports/figures/faz1_*.png
```

## Yönetici özeti

| # | Bulgu | Kanıt (ölçüm) | Etki |
|---|---|---|---|
| 1 | **Döngüsel değerlendirme (hedef sızıntısı):** Doğru cevap "aynı sektör mü?". Canlı sıralama (`match_candidate.py`) ise adayın altın `Category` alanından türetilen sektör çarpanını kullanıyor. | **Rastgele** skorlar × `match_candidate.py`'nin sektör çarpanı (0,85), `test.py` metriğiyle: **P@10 = %99,84, Top-1 = %100, NDCG (test.py) = %100** | Canlı sıralama bu metrikle ölçülürse metrik modeli değil, etiketi ölçer. Rehberdeki %67,87 / %79,78 / %86,93 ise `test.py`'nin çarpansız (ham benzerlik) Top-10 döngüsünden gelir; bu sayıları döngüsellik değil, NDCG'de yerel IDCG (bulgu 2, E1), P@10 ve Top-1'de zayıf "aynı sektör" alaka tanımı (E5) şişirir |
| 2 | **Yerel IDCG hatası:** `calculate_ndcg` IDCG'yi yalnızca dönen 10 sonuçtan hesaplıyor | `[2,0,0,…,0]` → test.py **1,000**, doğru NDCG **0,220**. BM25'te test.py %84,4, doğrusu %68,1 | NDCG sistematik olarak şişiyor |
| 3 | **CMQ kalite matrisi modelin kendi skorunu içeriyor** (%40 ağırlık) | Aynı rastgele sıralayıcı: kendi skoruyla kurulan CMQ'da 23,30, bağımsız skorla kurulanda 19,98 (+3,3 puan) | "Kademeli NDCG" kendini ödüllendiriyor |
| 4 | **Eğitim etiketleri bir modelin benzerlik skorundan türetilmiş** | SOFT_NEG: `label = 0,250·sim + 0,162`, **R² = 0,999**; SOFT_POS: R² = 0,982 | Model, öğretmenin yanlılığını regresyonla kopyalıyor |
| 5 | **Etiket belirteci girdide (kısayol öğrenme):** Eğitim ilan metinlerinin %100'ünde `Sektör: IT_SOFTWARE` gibi etiket belirteci var | Pozitif çiftlerin %93,95'i aynı sektörden; negatif çiftlerin %81,26'sı (HARD_NEG'lerin %100'ü) farklı sektörden | Model "sektör belirtecini eşle" kısayolunu öğrenir; test/çıkarım metninde belirteç olmadığından kısayol testte doğrudan işlemez (bulgu 7), ancak sektör eşleşmesine dayanan test metriği etiketlerden öğrenilen sektör yanlılığını ödüllendirir |
| 6 | **"HARD_NEG"ler aslında kolay negatif** | %100 farklı sektör, ortalama sim = −0,05 | Kontrastif anlamda zor negatif yok (benzerlikle seçilmiş görünen SOFT_NEG'ler yumuşak hedefle regresyona giriyor) |
| 7 | **Train/serve uyumsuzluğu:** Eğitimdeki ilan metni `Sektör:` ve `Pozisyon Seviyesi:` alanlarını içeriyor, test/çıkarım metni içermiyor | `sbertmodel.py` / `train.csv` ile `test.py` / `match_candidate.py` karşılaştırması | Dağılım kayması |
| 8 | **Şablon sızıntısı (entity değil):** ID kesişimi 0; neredeyse-kopya (cos ≥ 0,8) 0. Ancak test CV'lerinin **%54'ü** en az bir uzun cümleyi train CV'leriyle **birebir** paylaşıyor, **%30,9**'unun unvanı train'de birebir var | `reports/faz1_audit.json → C_eski_split` (bkz. bölüm C); cos dağılımı: `reports/figures/faz1_test_train_max_cos.png` | Ezber riski; yeni bölmede unvan grupları ayrık |
| 9 | Uydurma ve **tutarsız** çarpanlar | Sektör: rehber 0,50, `match_candidate.py` 0,85, `test.py` 'soft_rerank'te hiç yok; yaka: 0,0 ve 0,75; kıdem: 0,95^(fark−1) ve rehberde "0,70–1,00" | Literatür karşılığı yok, dosyalar arasında bile tutarsız |
| 10 | Doküman–veri tutarsızlığı | Rehberde "10042 = Kıdemli Java Geliştirici" yazıyor; veride 10042 **"Kontrolör"** (OFFICE_ADMIN_MGMT). Rehberde "boş açıklamalı Distribütör ilanı" geçiyor; SQL'de boş açıklama 0, "Distribütör ve Bayi Yöneticisi" 1.466 karakter | Vaka analizleri yeniden üretilemiyor |
| 11 | Güvenlik / taşınabilirlik | Kodda açık veritabanı parolası (`postgresql://postgres:denemeamacli@localhost`), sabit Windows yolları | Yeniden üretilebilirlik ve güvenlik açığı |

Ayrıca **BM25 (F5)**, eski metrik tanımlarıyla aynı 450 test CV'sinde **P@10 = %66,0, Top-1 = %76,4, NDCG
(test.py) = %84,4** alıyor. Rehberde ince ayarlı e5-small için verilen değerler sırasıyla %67,87, %79,78 ve %86,93.
Rehberdeki sayılar sektör çarpanı olmadan (ham benzerlikle) üretildiği için karşılaştırma çarpansız BM25 ile yapılmıştır.
Yani eski metrikle ince ayarlı model, eğitimsiz bir anahtar kelime aramasından ancak 1,9–3,3 puan iyi görünüyor.

---

## A. Veri envanteri ve profil

### A.1 `veriseti/tum_ilanlar.sql` → yapılandırılmış tablo
* 3.000 `INSERT` ifadesi PostgreSQL olmadan, karakter düzeyinde bir ayrıştırıcıyla okundu (`''` kaçışı ve
  `'…'::jsonb` desteklenir). `raw_data` JSON'u düz sütunlara açıldı. Çıktı: `work/tum_ilanlar.parquet` / `.csv`.
* `industry == raw_data.category` %100 tutarlı (`faz1_audit.json → A_profil.job_industry_raw_category_esit`);
  `title == raw_data.title` de %100 tutarlı *(ayrıca kontrol edildi: `work/tum_ilanlar.parquet` üzerinde
  `title == raw_title`; betik çıktısında yok)*. Toplam **2.973 tekil unvan** var (27 tekrar).

| Alan | Dağılım |
|---|---|
| Sektör (26) | INDUSTRIAL_MANUFACTURING 435, IT_SOFTWARE 410, CONSTRUCTION 320, HEALTHCARE 207, FINANCE_ACCOUNTING 194, SALES 145, MARKETING_MEDIA 143, CHEF_HORECA 116, HR 102, SCIENCE_RND 97, … BEAUTY_CARE 8 |
| position_level (17 farklı yazım!) | SPECIALIST 762, SENIOR 748, MID 570, ENTRY 351, LEAD 160, ENTRY_LEVEL 131, MANAGER 85, UNSPECIFIED 59, MID_LEVEL 56, INTERN 41, … |
| Çalışma şekli | OnSite 2.026, Hybrid 800, Remote 174 |
| İş tipi | FullTime 2.845, PartTime 126, Internship 12, Freelance 8, Daily 8, Contract 1 |
| Şehir (ilk 5) | İstanbul 1.252, Kocaeli 379, Ankara 321, Bursa 297, İzmir 211 |
| Dil | TR 2.439, alan yok 500, EN 61 |
| Açıklama uzunluğu | ort. 1.049 kr, min 63, maks 3.399; 200 karakterden kısa 25 ilan; boş açıklama **0** |
| Maskeleme | 1.837 açıklamada `[şehir]` maskesi |

> **Not:** `position_level` 17 farklı yazımla geliyor (ENTRY / ENTRY_LEVEL / NEW_GRADUATE, MID / MID_LEVEL /
> SPECIALIST …). Yeni sistemde UNSPECIFIED dışındaki 16 yazım 0–5 ölçeğine eşlenir
> (`match_candidate_sota.JOB_LEVEL_MAP`); UNSPECIFIED (59 ilan) eşlenmez, kıdem bilinmiyor sayılır
> ("Belirtilmemiş"; sıralama özniteliklerinde `job_seniority_missing` = 1).

### A.2 `veriseti/tum_cvler.csv`
* 3.043 CV (ID 10000–21043, tekil), 2.486 tekil unvan. Yaka: BEYAZ 2.035 / MAVİ 1.008. Lokasyon: 34 şehir
  (İstanbul 909, Ankara 642, İzmir 274, …).
* `cv_text` ortalama 1.100 karakter (410–2.129). `cv_json` içindeki `about` alanı CV'lerin %14,7'sinde (447/3.043;
  boşluklar normalize edilerek alt-dizgi kontrolü, `faz1_audit.json → A_profil.cv_about_cv_text_farkli_pct`)
  `cv_text`'teki özetle **aynı değil**. Birincil kaynak olarak `cv_text` kullanıldı.
* Eksikler: deneyim bölümü yok 51, sertifika yok 545. Maskeler: `[TARİH]` 1.952, `[ŞİRKET]` 1.928, `[ÜNİVERSİTE]` 304,
  `[OKUL]` 312 CV'de. 285 CV ağırlıklı olarak İngilizce.
* Metinden çıkarılan kıdem (0–5): önce unvan anahtar kelimelerinden, yoksa deneyim yılından, o da yoksa metindeki
  "son sınıf / öğrencisiyim / staj imkânı" gibi ifadelerden (0) hesaplanır; hiçbiri yoksa varsayılan 2 atanır
  (3.043 CV'de sırasıyla 753 / 577 / 24 / 1.689; *ayrıca kontrol edildi*). Etiket kullanılmaz
  (`common.parse_seniority_text`).
* **Sektör dengesizliği:** CUSTOMER_SERVICE 8 CV'ye karşılık 53 ilan (oran 0,15). Diğer sektörlerde CV/ilan
  oranı 0,5–1,5 arasında. Tam tablo `faz1_audit.json → A_profil.sektor_cv_ilan_orani` içinde,
  grafik `reports/figures/faz1_sektor_cv_ilan.png`.

### A.3 `train.csv` / `test.csv` / `train_groups_summary.json`
* `train.csv`: 25.906 çift, 2.593 CV, 2.853 ilan. `match_type`: HARD_NEG 12.836, SOFT_POS 6.485, SOFT_NEG 5.100,
  EXACT_POS 1.485.
* `test.csv`: 450 CV (yalnız CV; ilan çifti yok).
* `train_groups_summary.json` bölmeyi değil, yalnız train'deki 2.593 CV'yi ardışık ID aralıklarıyla 10 grupta
  özetler (`group_count` = 10; 283/284 CV'lik 9 grup + 43 stajyer CV). 450 test CV'sinin 448'i bu aralıkların
  içindedir *(ayrıca kontrol edildi: `veriseti/test.csv`)*; yani gruplar bölme birimi değildir, eski bölme grup
  düzeyinde değil, yalnızca CV kimliği düzeyinde ayrıktır.

## B. Şablon tekrarı

| Ölçüm | Değer |
|---|---|
| ≥ 5 CV'de birebir geçen (≥ 8 kelimelik, unvan maskeli) cümle sayısı | **113** |
| Bu şablon cümlelerinden en az birini içeren CV oranı | **%38,3** |
| En sık şablon | "takım çalışmasına uyumlu dinamik çalışma ortamlarına hızla adapte olabilen ve sorumluluk bilinci yüksek bir uzmanım" (**218 CV**) |
| Neredeyse-kopya CV çifti (unvan maskeli TF-IDF cos ≥ 0,8 / 0,9) | 1 / 0 |

Sonuç: Kopya CV yok, ancak metinlerin önemli bir kısmı **sentetik şablon cümleleri** taşıyor. Bunlar meslek
bilgisi taşımayan "gürültü" cümleleri. Model bunlara değil, meslek içeriğine odaklanmalı. Yeni sistemin
yapılandırılmış metin şablonu (`build_cv_struct`) bu yüzden **Pozisyon** ve metinden çıkarılan **Kıdem** bilgisini
başa ekler, **Yetkinlikler**'i de Özet'in önüne alır (sıra: Pozisyon, Kıdem, Konum, Yetkinlikler, Özet, Deneyim,
Eğitim, …). Şablon cümlelerini taşıyan Özet ise Deneyim'den önce kalır.

## C. Eski train/test ayrımında sızıntı

| Ölçüm | Değer | Yorum |
|---|---|---|
| CV ID kesişimi (train ∩ test) | **0** | Kimlik düzeyinde (entity) sızıntı **yok**: bu iddia doğru |
| Test CV'lerinin train'de birebir aynı unvanı olanlar | **%30,89** | Unvan düzeyinde örtüşme |
| Test→train en yakın kosinüs (unvan maskeli TF-IDF) medyan / p90 | 0,328 / 0,628 | |
| Train'de cos ≥ 0,8 ikizi olan test CV'si | **%0** | Neredeyse-kopya sızıntısı yok |
| Train ile ≥ 8 kelimelik cümleyi birebir paylaşan test CV'si | **%54,0** | **Şablon sızıntısı** |
| Kategori dağılımı farkı (Jensen–Shannon) | 0,0002 bit | Bölme dağılım olarak dengeli |
| Test CV'leri için "doğru" sayılan (aynı sektör) ilanların train'de **pozitif** görülme oranı | **%70,67** | Değerlendirmede ödüllendirilen ilanların çoğu eğitimde pozitif etiketle ezberlenmiş |

**Matematiksel sonuç:** $\text{ID}_{train} \cap \text{ID}_{test} = \emptyset$ olduğu için klasik entity leakage yok.
Ancak (i) test sorgularının %54'ü train ile ≥ 8 kelimelik en az bir cümleyi birebir paylaşıyor (B'deki ≥ 5 CV'lik
şablon cümlelerle sınırlanınca %36,7; *ayrıca kontrol edildi*) ve (ii) test metriğinin "doğru" saydığı
ilanların %70,7'si eğitimde pozitif etiketle görüldü. Yani eski protokol, *aynı ilanlara aynı şablonlu CV'ler*
üzerinden **transdüktif** (eğitimde görülmüş öğelerle) bir değerlendirme yapıyor. Yeni protokolde:
(1) normalize unvan ve neredeyse-kopya grupları bölümler arasında ayrıktır (unvan kesişimi = 0; `reports/faz2_veri_ozeti.json`),
(2) ilanların %10'u (*cold*) hiçbir eğitim sinyalinde kullanılmaz ve ayrıca raporlanır.

## D. `train.csv` etiketlerinin kaynağı

| match_type | n | etiket ort. | sim ort. | Pearson(etiket, sim) | aynı sektör |
|---|---|---|---|---|---|
| EXACT_POS | 1.485 | 0,971 | 0,869 | 0,545 | %95,7 |
| SOFT_POS | 6.485 | 0,647 | 0,696 | **0,991** | %93,6 |
| SOFT_NEG | 5.100 | 0,294 | 0,525 | **1,000** | %65,9 |
| HARD_NEG | 12.836 | 0,000 | −0,051 | — | **%0,0** |

Doğrusal uyum: SOFT_NEG için `label = 0,2502·sim + 0,1624` (**R² = 0,9991**), SOFT_POS için
`label = 0,4709·sim + 0,3196` (R² = 0,9819).

**Sonuç:** "Yumuşak" etiketler, bir modelin kosinüs benzerliğinin doğrusal dönüşümü. `CosineSimilarityLoss`
ile bu hedeflere regresyon yapmak, öğretmen modelin hatalarını öğrenciye kopyalamak (*confirmation bias /
self-training drift*) demek. "HARD_NEG" adlı örneklerin tamamı farklı sektörden ve neredeyse sıfır benzerlikli,
yani literatürdeki anlamıyla **kolay** negatif. Benzerliğe göre seçilmiş görünen tek negatifler SOFT_NEG'ler: aday
başına tam 2 (2.550 CV), sim ≤ 0,55 eşiğinin hemen altında (medyan 0,541), %65,9'u aynı sektör; SOFT_POS'lar 0,60'ta
başlıyor (EXACT_POS'ların yalnızca 17/1.485'i 0,60'ın altında) *(ayrıca kontrol edildi: `veriseti/train.csv`)*. Ancak SOFT_NEG'ler `CosineSimilarityLoss` ile 0,23–0,30 yumuşak
hedeflere regresyonla öğretildiği ve batch-içi negatif olmadığı için kontrastif anlamda zor negatif
(Karpukhin vd., 2020; Xiong vd., 2021) işlevi görmüyor.

Ek olarak, eğitim ilan metinlerinin **%100'ü** `Sektör: <ETİKET>` belirtecini içeriyor. Pozitiflerin %94'ü aynı
sektörden, negatiflerin %81,3'ü farklı sektörden (HARD_NEG'lerin %100'ü, SOFT_NEG'lerin yalnızca %34,1'i) olduğu
için model sektör belirtecini eşleyen bir **kısayol** öğrenir
(Geirhos vd., 2020). Test metriği de sektör eşleşmesi olduğundan etiketlerden öğrenilen sektör yanlılığı
ödüllenir. Ancak değerlendirme ve çıkarım ilan metni (`test.py`, `match_candidate.py`) bu belirteci içermez:
belirteç kısayolu testte doğrudan kullanılamaz ve ayrıca bir **train/serve uyumsuzluğu** var.

## E. `test.py` değerlendirmesindeki metodolojik hatalar

### E1. Yerel IDCG (NDCG şişirmesi)
```python
ideal_list = sorted(relevance_list, reverse=True)   # relevance_list = YALNIZCA dönen top-10
```
Literatürdeki NDCG@k'da IDCG, sorgu için **havuzdaki tüm** alakalı belgelerin ideal sıralamasından hesaplanır
(Järvelin & Kekäläinen, 2002). `test.py`'de IDCG sadece dönen 10 sonuçtan hesaplanıyor. Bu yüzden listeye kaç alakalı
belge girdiği değil, yalnızca **dönenlerin kendi aralarındaki sırası** ölçülüyor:

| Liste (rel) | test.py NDCG | Doğru NDCG (havuzda 50 alakalı) |
|---|---|---|
| `[2,0,0,0,0,0,0,0,0,0]` | **1,000** | 0,220 |
| `[0,0,0,0,0,0,0,0,0,2]` | 0,289 | — |

BM25 sıralamasında (450 test CV'si): test.py NDCG **%84,38**, doğru NDCG **%68,09**.

### E2. Sektör çarpanı ile döngüsel değerlendirme (hedef sızıntısı)
Alaka tanımı: `rel = 2 if Category(aday) == industry(ilan)`.
Canlı sıralama (`match_candidate.py`): `skor = ham × … × ind_m`, burada `ind_m = 1` (aynı sektör) ya da farklı
sektörde `0,85` (kod; eski rehberin metninde `0,50`). `ind_m`, **etiketin kendisinden** (adayın altın `Category`
alanı) hesaplanıyor.

**Analitik kanıt.** Varsayım: kosinüsler dar bir aralıkta toplanır; burada $s \in [a,b] \approx [0{,}70; 0{,}90]$ alınmıştır
(taban E5 modellerinde kosinüslerin, düşük InfoNCE sıcaklığı (0,01) nedeniyle yaklaşık 0,7–1,0 bandında toplandığı
bilinir; üst sınır 0,90 simülasyonun kendi varsayımıdır). Eski ince ayarlı modelin aralığı
ölçülmedi (ağırlıkları yok); `CosineSimilarityLoss` HARD_NEG çiftlerini 0'a ittiği için gerçek aralık daha geniş
olabilir (eski rehberdeki 0,9049 ham benzerlik zaten $b$'nin üstünde); bu durumda aşağıdaki analitik koşul garanti
değildir.
Çarpan $c$ için $c\cdot b < a$ ise hiçbir farklı sektör ilanı herhangi bir aynı sektör ilanının önüne geçemez. Aday
başına aynı sektör ilan sayısı medyan 194, minimum 8 olduğundan P@10 ≈ 1 olur. $c=0{,}50$ için
$0{,}45 < 0{,}70$ ✔. $c = 0{,}85$ için koşul kesin sağlanmaz ($0{,}765 > 0{,}70$), ancak pratikte yine sağlanır (aşağıda).

**Ampirik kanıt** (450 test CV'si, sıralayıcı = **U(0,70; 0,90) rastgele sayılar**, model YOK):

| Sıralayıcı | P@10 | Top-1 | NDCG (test.py) | NDCG (doğru) |
|---|---|---|---|---|
| Rastgele | 7,49 | 3,11 | 22,03 | 6,68 |
| **Rastgele × sektör çarpanı 0,85** (`match_candidate.py`) | **99,84** | **100,00** | **100,00** | 99,95 |
| **Rastgele × sektör çarpanı 0,50** (rehber) | **99,91** | **100,00** | **100,00** | 100,00 |
| BM25 | 66,02 | 76,44 | 84,38 | 68,09 |
| BM25 × sektör çarpanı 0,50 | 92,22 | 96,00 | 97,61 | 93,17 |

**Hiçbir bilgi içermeyen rastgele bir sıralayıcı, sektör çarpanıyla %100 Top-1 alıyor.** Böyle bir metrik modelin
başarısını değil, etiketin sıralamaya sızdırıldığını ölçer. Bu, Kaufman vd. (2012, *Leakage in data mining*)
tanımıyla **hedef sızıntısıdır**.

**Kapsam notu.** Bu kanıt, eski rehberde raporlanan sayıların nasıl üretildiğini değil, canlı sıralamanın
(`match_candidate.py`) "aynı sektör" metriğiyle doğrulanamayacağını gösterir. `test.py` raporlanan Top-10 metriklerini
`np.argsort(similarity_matrix[i])` ile, yani ham benzerlikle ve sektör çarpanı olmadan hesaplar (`soft_rerank`'te de
sektör çarpanı yoktur). Raporlanan %67,87 / %79,78 / %86,93'ü şişiren, NDCG için yerel IDCG (E1), P@10 ve Top-1 için
zayıf alaka tanımıdır (E5); nitekim çarpansız BM25 aynı metrikle %66,02 / %76,44 / %84,38 alır.

### E3. CMQ "Bileşik Kalite" matrisi modelin kendi skorunu içeriyor
`compute_all_qualities_matrix`: `Q = yaka × (0,40·s_model + 0,25·sektör + 0,20·kıdem + 0,15·lokasyon)`.
Doğru cevabın %40'ı, değerlendirilen modelin kendi skoru. Model sıralamasını kendi skoruna göre yaptığından bu
metriği kendiliğinden yükseltir. Aynı rastgele sıralayıcı için: kendi skoruyla kurulan CMQ'da kademeli NDCG
**23,30**, bağımsız bir rastgele skorla kurulan CMQ'da **19,98** (**+3,32** puan, yalnızca kendine referanstan).
Gerçek bir modelde, skor dağılımı keskinleştikçe bu fark büyür.

### E4. Regex yaka sınıflandırıcısı hem filtrede hem alakada
`run_evaluation(filter_mode="collar")`: sıralama `collar_mult` ile cezalandırılıyor ve **aynı** regex sonucu
alakanın parçası (`rel = sektör eşit VE yaka eşit`). Rastgele skorla P@5 %6,27'den %9,38'e çıkıyor; artış modelden
değil, yaka filtresinden geliyor. Skor 0,70–0,90 aralığında olduğundan 0,75 çarpanı (0,90 × 0,75 = 0,675 < 0,70)
top-5'i tamamen adayla aynı yakadaki ilanlara kısıtlar; ilan yakası filtrede ve alakada aynı regex'ten geldiği için
alakaya eklenen "yaka eşit" koşulu kendiliğinden sağlanır ve sonuca katkısı sıfırdır (filtre + yalnız sektör alakası
da %9,38 verir; filtresiz "sektör ve yaka" alakası %4,49'a düşer; *ayrıca kontrol edildi*, aynı tohumla). Artış,
aynı yakadaki ilanlar arasında aynı sektör oranının daha yüksek olmasından kaynaklanır. Ayrıca regex yaka sınıflandırıcısı, veri setindeki yaka etiketiyle CV'lerin
%1,97'sinde çelişiyor.

### E5. Ek gözlemler
* **Metrik kapsamı:** "Doğru" = aynı sektör. Bir *Java stajyeri* için IT sektöründeki *CTO* ilanı da "doğru" sayılıyor.
  Meslek, kıdem ve lokasyon uyumu ölçülmüyor.
* **Eğitim:** `CosineSimilarityLoss` + batch 8 ile batch-içi negatif yok. Kontrastif öğrenmenin
  (van den Oord vd., 2018; Henderson vd., 2017) temel avantajı kullanılmıyor.

## F. Sonuç: yeni sistemin tasarım kararlarına etkisi

| Eski hata | Yeni sistemdeki karşılığı |
|---|---|
| Etiketten türetilen çarpanlar (E2) | Çarpan yok. Sıralamayı **LambdaMART** öğrenir. Sektör/yaka uyumu yalnızca **train'de eğitilmiş metin sınıflandırıcılarının tahmin olasılıklarıdır**; öznitelikler yalnızca CV metninden ve ilanın kendi meta verisinden üretilir. `assert_no_gold_columns` buna ek olarak sütun adı düzeyinde bir şema korumasıdır (altın/etiket sütunu adlarını reddeder, değerleri denetlemez) |
| Yerel IDCG (E1) | IDCG tüm havuzdaki derecelerden (`match_candidate_sota.ndcg_at_k`) |
| Kendine referanslı CMQ (E3) | Ground truth, modelden bağımsız **ISCO-08 meslek kodu + uzmanlık + yaka + kıdem + lokasyon** (grade 1 için ana sektör) kurallarından üretilir. Tüm 3.043 × 3.000 çift için tam yargı vardır |
| Model-türevi etiketler (D) | Etiketler hiçbir sıralama modelinin skorundan türetilmez. LLM destekli meslek anotasyonu + IAA (Cohen κ) + çift düzeyi doğrulama |
| Kolay "hard negative"ler (D) | 4 tür gerçek zor negatif: leksikal tuzak, kıdem/yaka tuzağı, BM25, qrels-filtreli yoğun madencilik. Hepsi qrels'e göre grade 0 (kıdem/yaka tuzağı ≤ 1); en benzer grade-0 ilanlar seçildiği için bir kısmı gerçekte yakın-alakalı olabilir (kural kaynaklı yanlış negatif nadir ama sıfır değil) |
| Etiket belirteci kısayolu (D) | CV metninde altın etiket yok. İlan metnindeki "Sektör" ilanın kendi meta verisi; ikili metriklerin alaka eşiği (grade ≥ 2) mesleğe (ISCO-08 birim grubu / uzmanlık) dayanır. Sektör yalnızca grade 1'de rol oynar (kısmi alaka: yaka uyumu + aynı ana sektör) ve kademeli NDCG@10'a kazanç 1 ile girer |
| Şablon sızıntısı (C) | Normalize unvan ve neredeyse-kopya grupları bölümler arasında ayrık; ayrıca %10 soğuk-başlangıç ilan havuzu |
