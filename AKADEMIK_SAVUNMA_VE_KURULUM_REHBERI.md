# CV ↔ İş İlanı Eşleştirme — Akademik Savunma ve Kurulum Rehberi

> Bu doküman dört bölümden oluşur: (1) Colab A100 kurulumu ve çalıştırma, (2) eski sistemin bilimsel eleştirisi,
> (3) yeni mimarideki her bileşenin teorik ve matematiksel gerekçesi, (4) jüriden gelebilecek en zor 10 soru ve
> cevapları. Rehberdeki her sayı depodaki scriptlerle yeniden üretilebilir. Kaynakça en sondadır.

**Teslim edilen dosyalar**

| Dosya | Ne işe yarar |
|---|---|
| `CV_Ilan_SOTA_Egitim_ve_Test_A100.ipynb` | Colab A100'de uçtan uca eğitim, test, ablation, hata analizi ve dışa aktarım |
| `data_bundle.zip` | Sızıntısız, temiz veri paketi (Colab'a tek dosya olarak yüklenir) |
| `match_candidate_sota.py` | Eğitim ve çıkarımın ortak çekirdeği + CLI/API (`--serve` ile FastAPI) |
| `reports/FAZ1_ADLI_DENETIM_RAPORU.md` | Eski sistemin sayısal adli denetimi |
| `reports/FAZ2_VERI_RAPORU.md` | Yeni veri setinin, etiketlerin ve sızıntı kontrollerinin raporu |
| `annotation/` | Meslek anotasyon kılavuzu, ham anotasyonlar, uyum (IAA) ölçümleri |
| `scripts/01…08_*.py` | Verinin ve notebook'un tamamını yeniden üreten scriptler |

---

## BÖLÜM 1 — Google Colab A100'de adım adım çalıştırma (teknik bilgi gerekmez)

1. **Dosyaları indirin.** GitHub'dan `CV_Ilan_SOTA_Egitim_ve_Test_A100.ipynb` ve `data_bundle.zip` dosyalarını
   bilgisayarınıza indirin (dosyaya tıklayın → sağ üstte *Download raw file*).
2. **Colab'ı açın.** <https://colab.research.google.com> → *Dosya → Not defteri yükle* → `.ipynb` dosyasını seçin.
3. **A100'ü seçin.** *Çalışma zamanı → Çalışma zamanı türünü değiştir* → *Donanım hızlandırıcı: **A100 GPU***
   (Colab Pro / Pro+ / kredi gerekir) → *Kaydet*. İsterseniz *Yüksek RAM*'i de açın.
4. **Veriyi yükleyin.** İki yol var:
   * Sol menüde 📁 *Dosyalar* paneline `data_bundle.zip` dosyasını sürükleyin (`/content/data_bundle.zip` olur), **veya**
   * hiçbir şey yapmayın: 2. bölümdeki hücre çalışınca bir **"Dosya seç"** düğmesi çıkar, oradan seçin.
5. **Çalıştırın.** *Çalışma zamanı → Tümünü çalıştır*. İlk hücre kütüphaneleri sabit sürümlerle kurar (~2 dk).
   Modeller (BGE-M3 ≈ 2,3 GB, bge-reranker-v2-m3 ≈ 2,3 GB, e5-small ≈ 0,5 GB) Hugging Face'ten otomatik iner.
6. **Bekleyin.** A100-40GB'da toplam süre yaklaşık **60–90 dakika**:

   | Bölüm | Yaklaşık süre |
   |---|---|
   | Kurulum + veri | 3 dk |
   | Baseline 1–2 (BM25, zero-shot) | 2 dk |
   | Baseline 3 (eski yaklaşımın yeniden eğitimi, 3 epoch) | 8–12 dk |
   | Sıcaklık taraması (3 × kısa eğitim) | 10–15 dk |
   | Aşama 1 bi-encoder (3 epoch, erken durdurma) | 15–25 dk |
   | Aşama 2a cross-encoder (2 epoch) | 10–15 dk |
   | Aşama 2b LambdaMART (Optuna 40 deneme) | 2–4 dk |
   | Ablation, hata analizi, dışa aktarım | 5 dk |

7. **Sonuçları alın.** Son hücreler `export_model_bundle.zip`'i otomatik olarak indirir. Grafikler, CSV tabloları ve
   `metrics.json` bu paketin içindedir (`figures/`, `ablation_test.csv`, `cold_start_test.csv`).
8. **Canlıda kullanım (arkadaşınız için):**
   ```bash
   unzip export_model_bundle.zip && cd export_model_bundle
   pip install sentence-transformers==6.1.0 transformers==5.17.0 lightgbm==4.7.0 faiss-cpu scikit-learn pandas pyarrow joblib
   python match_candidate_sota.py --bundle . --cv-id 20951                  # veri setindeki bir aday
   python match_candidate_sota.py --bundle . --cv-file aday.txt --headline "Baklava Ustası"
   python match_candidate_sota.py --bundle . --serve                        # POST /match {cv_text, headline, top_k}
   ```

**Sık karşılaşılan sorunlar**

| Belirti | Çözüm |
|---|---|
| `CUDA out of memory` | 1. bölümdeki `CFG` içinde `BI_BATCH=32`, `BI_MINI_BATCH=8`, `CE_BATCH=4` yapın |
| A100 yerine T4/L4 geldi | Çalışır ama yavaştır. L4'te bf16 vardır; T4'te `BF16=False` olur, süre 3–4 katına çıkar |
| Kurulum sonrası "restart runtime" uyarısı | *Çalışma zamanı → Yeniden başlat*, ardından tekrar *Tümünü çalıştır* (kurulum hızlıca atlanır) |
| Hugging Face indirmesi yavaş | Colab ağına bağlıdır. Model adları `CFG` içinde değiştirilebilir (ör. `intfloat/multilingual-e5-large`) |
| Hızlı bir deneme isteniyor | `CFG`'de `BI_EPOCHS=1`, `TUNE_TEMPERATURE=False`, `LTR_TRIALS=10` yapın (~25 dk) |

**Doğrulama:** Notebook bu depoda CPU üzerinde `DRY_RUN` modunda (küçük, rastgele başlatılmış BERT'lerle)
uçtan uca çalıştırılmıştır: `python scripts/08_dry_run_notebook.py` → `reports/notebook_dogrulama.json`. Bu çalışma
yalnızca **kodun hatasız çalıştığını** kanıtlar. Rastgele modellerin metrikleri anlamsızdır; gerçek metrikler
A100 çalıştırmasından gelir.

---

## BÖLÜM 2 — Eski projedeki metodolojik hataların bilimsel eleştirisi

Tüm sayılar `reports/FAZ1_ADLI_DENETIM_RAPORU.md` ve `scripts/03_forensic_audit.py` kaynaklıdır.

### 2.1 Döngüsel değerlendirme = hedef sızıntısı (target leakage)
Eski formül: $\text{Skor} = s_{\text{ham}} \times c_{\text{yaka}} \times c_{\text{kıdem}} \times c_{\text{lok}} \times c_{\text{sektör}}$.
Burada $c_{\text{sektör}}=\mathbb{1}[\text{Cat}(cv)=\text{Ind}(ilan)] + 0{,}85\,\mathbb{1}[\ne]$.
Başarı ölçütü ise $\text{rel}=2\cdot\mathbb{1}[\text{Cat}(cv)=\text{Ind}(ilan)]$.
**Sıralama fonksiyonu, doğru cevap fonksiyonunun kendisini içeriyor.** Kaufman vd. (2012) bunu *leakage* olarak
tanımlar: tahmin anında meşru olarak bulunmayacak, hedefle aynı bilgiyi taşıyan bir girdinin modele verilmesi.

**Analitik kanıt.** E5 kosinüsleri dar bir bantta toplanır, $s\in[a,b]\approx[0{,}70;0{,}90]$. Farklı sektördeki bir
ilanın skoru en fazla $c\cdot b$, aynı sektördeki bir ilanınki en az $a$ olur. $c\cdot b<a$ ise hiçbir farklı sektör
ilanı üst sıralara çıkamaz ve aday başına en az 10 aynı sektör ilanı varsa $\text{P@10}=1$ olur. (Aday başına aynı
sektör ilan sayısının medyanı 194'tür.)

**Ampirik kanıt.** Hiçbir bilgi içermeyen **rastgele** skorlar sektör çarpanıyla çarpıldığında, 450 test CV'sinde
**P@10 = %99,84, Top-1 = %100** elde ediliyor. Metrik modelin değil, sızdırılan etiketin başarısını ölçüyor.

### 2.2 NDCG'nin yanlış hesaplanması (yerel IDCG)
$\mathrm{NDCG@}k=\mathrm{DCG@}k/\mathrm{IDCG@}k$ tanımında IDCG, sorgunun **havuzdaki tüm** alakalı belgelerinin ideal
sıralamasıdır (Järvelin & Kekäläinen, 2002). Eski kod IDCG'yi yalnızca dönen 10 sonucun kendisinden hesaplıyor.
Bu yüzden `[2,0,0,…,0]` listesi NDCG = 1,0 alıyor; doğrusu (50 alakalı belgeli havuzda) 0,22. BM25 sıralamasında
bu hata NDCG'yi %68,1'den %84,4'e şişiriyor.

### 2.3 Kendine referanslı "Bileşik Kalite" (CMQ)
$Q = \text{yaka}\cdot(0{,}40\,s_{\text{model}} + 0{,}25\,\text{sektör} + 0{,}20\,\text{kıdem} + 0{,}15\,\text{lok})$.
Doğru cevabın %40'ı, değerlendirilen modelin kendi skoru. Model kendini ölçüyor; rastgele bir sıralayıcı bile yalnızca
bu kendine referans sayesinde +3,3 puan kazanıyor.

### 2.4 Model-türevi (pseudo) etiketler ve kısayol öğrenme
* Eğitimdeki "yumuşak" etiketler bir modelin kosinüs skorunun doğrusal dönüşümü (SOFT_NEG: R² = 0,999). Bir modeli
  başka bir modelin skorlarına regresyonla eğitmek, öğretmenin hatalarını öğrenciye kopyalar.
* Eğitim ilan metinlerinin %100'ü `Sektör: <ETİKET>` belirtecini içeriyor; pozitiflerin %94'ü aynı sektörden,
  negatiflerin %100'ü farklı sektörden. Model "etiket belirtecini eşle" kısayolunu öğrenir (Geirhos vd., 2020).
  Değerlendirme de sektör eşleşmesi olduğu için kısayol ödüllendirilir.
* "HARD_NEG" etiketli 12.836 örneğin %100'ü farklı sektörden ve ortalama benzerliği −0,05. Bunlar **kolay** negatif.
  Zor negatif yokluğu, Karpukhin vd. (2020) ve Xiong vd. (2021) tarafından gösterilen başlıca performans
  kayıplarından biridir.
* `CosineSimilarityLoss` + batch 8 → batch-içi negatif yok, kontrastif sinyal yok.

### 2.5 Uydurma ve tutarsız çarpanlar
0,50 / 0,70 / 0,85 / 0,95 katsayılarının kaynağı yok: bir veri üzerinden öğrenilmemiş, bir doğrulama kümesinde
ayarlanmamış ve dosyalar arasında bile tutarsız (rehberde sektör 0,50, `match_candidate.py`'de 0,85, `test.py`'de
hiç yok). Birden çok sinyali birleştirmenin literatürdeki karşılığı **öğrenilen sıralama** (Learning-to-Rank;
Liu, 2009; Burges, 2010) ya da **rank füzyonu**dur (Cormack vd., 2009).

### 2.6 Bölme ve şablon sızıntısı
ID düzeyinde kesişim yok (eski iddia doğru). Ancak test CV'lerinin %54'ü uzun bir cümleyi train ile birebir paylaşıyor,
%31'inin unvanı train'de birebir mevcut ve test metriğinin "doğru" saydığı ilanların %71'i eğitimde pozitif etiketle
görülmüş. Yeni protokolde unvan grupları ayrıktır ve ilanların %10'u (*cold*) hiçbir eğitim sinyalinde kullanılmaz.

### 2.7 Metriğin ölçtüğü şey yanlış
"Doğru = aynı sektör" tanımında bir *Java stajyeri* için *CTO* ilanı da doğru sayılır. Meslek, kıdem, yaka ve lokasyon
uyumu ölçülmez. Yeni sistemde alaka **dört kademelidir** ve meslek koduna dayanır (Bölüm 3.2).

---

## BÖLÜM 3 — Seçilen bileşenlerin teorik ve matematiksel karşılığı

### 3.1 Problemin formülasyonu
Sorgu $q$ (CV), belge kümesi $\mathcal{D}$ (3.000 ilan), kademeli alaka $r(q,d)\in\{0,1,2,3\}$. Amaç, beklenen
$\mathrm{NDCG@10}$'u en büyükleyen bir sıralama fonksiyonu $f(q,d)$ öğrenmektir. Bu bir *bilgi getirimi* ve
*öğrenen sıralama* problemidir (Liu, 2009); ikili sınıflandırma değildir. Bu yüzden F1 değil, sıralama metrikleri
kullanılır.

### 3.2 Ground truth: modelden bağımsız, kademeli, tam yargılı qrels
* **Meslek ontolojisi:** Her CV ve ilan, ILO'nun **ISCO-08** sınıflamasının 4 haneli birim grubuna (ILO, 2012)
  ve onun altında kanonik bir **uzmanlık** etiketine (ESCO düzeyi inceliğinde) eşlendi. Ayrıca yaka ve 0–5 ölçekli
  kıdem atandı (`annotation/GUIDELINES.md`).
* **Anotasyon süreci:** LLM destekli uzman anotasyonu (Thomas vd., 2024; Faggioli vd., 2023), yazılı kılavuz ve ev
  kurallarıyla yapıldı. Ardından ISCO alt-ana gruplarına göre bir **konsolidasyon** turu, ayrıca öğelerin
  rastgele %10'u için **bağımsız ikinci anotasyon** yapıldı. Uyum Cohen'in κ katsayısıyla raporlanır (Cohen, 1960;
  kıdem için ağırlıklı κ, Cohen, 1968). Değerler `reports/FAZ2_VERI_RAPORU.md` içindedir.
* **Kademeler:**

  | Grade | Kural |
  |---|---|
  | 3 | aynı uzmanlık ∧ aynı yaka ∧ \|Δkıdem\| ≤ 1 ∧ (aynı şehir ∨ Remote) |
  | 2 | aynı ISCO-08 birim grubu ∧ aynı yaka ∧ \|Δkıdem\| ≤ 2 (kıdem/lokasyonda tolere edilebilir fark) |
  | 1 | aynı yaka ∧ (aynı ana sektör ∨ aynı ISCO grubu ama \|Δkıdem\| ≥ 3) |
  | 0 | diğer |

* **Tam yargı (complete judgments):** Kademeler kurallardan türetildiği için **tüm** 3.043 × 3.000 çift etiketlidir.
  TREC tarzı *pooling*'in yarattığı "yargılanmamış = alakasız" yanlılığı (Buckley & Voorhees, 2004) burada yoktur.
  Zor negatif madenciliğinde **yanlış negatif** riski de sıfıra iner: bir ilanın grade-0 olduğu bilinir, tahmin edilmez.

### 3.3 Sızıntısız bölme
* Sorgular (CV'ler) **ayrık** üç kümeye bölünür (70/15/15). Bölme `StratifiedGroupKFold` ile sektöre göre
  tabakalıdır. **Grup** = aynı normalize unvan ∪ neredeyse-kopya bileşeni (unvan maskeli TF-IDF cos ≥ 0,8).
  Bu yüzden test CV'lerinin hiçbirinin unvanı train'de bulunmaz.
* Vitrin adayları (Baklava ustası 20951, uçuş dispeçeri 10587, Java geliştiricisi 10511, Java stajyeri 21029)
  test kümesine sabitlenmiştir. Hata analizinde gösterilen her örnek eğitimde hiç görülmemiştir.
* İlanların %10'u (sektöre göre tabakalı) **cold** havuzudur: triplet, sentetik CV, cross-encoder listeleri ve
  LambdaMART eğitimi dahil hiçbir eğitim sinyalinde kullanılmaz. Ayrıca ayrı bir **soğuk-başlangıç** değerlendirmesi
  yapılır.
* Sentetik CV'ler yalnızca train'e girer. Başlığı bir val/test CV'siyle çakışan sentetik CV atılır.

### 3.4 Aşama 1a — Yoğun bi-encoder (DPR / Sentence-BERT)
Sorgu ve belge ayrı ayrı kodlanır: $s(q,d)=\cos(E(q),E(d))$ (Reimers & Gurevych, 2019; Karpukhin vd., 2020).
Belgeler önceden kodlanıp FAISS'te indekslenir. Çıkarım maliyeti $O(1)$ kodlama + en yakın komşu aramasıdır.

**Temel model: BGE-M3** (Chen vd., 2024). XLM-RoBERTa-large tabanlıdır, 100+ dil (Türkçe dahil) destekler ve 8.192
belirteçlik bağlam alır. Çok dilli MIRACL/MKQA kıyaslamalarında güçlüdür. Alternatif olarak `CFG` içinden
`multilingual-e5-large` seçilebilir (Wang vd., 2024).

**Kayıp: InfoNCE / Multiple Negatives Ranking Loss** (van den Oord vd., 2018; Henderson vd., 2017):
$$\mathcal{L}_{\text{MNRL}}=-\frac{1}{|B|}\sum_{i\in B}\log\frac{\exp(s(q_i,d_i^+)/\tau)}{\sum_{d\in\{d_j^+\}_{j\in B}\,\cup\,\mathcal{H}_i}\exp(s(q_i,d)/\tau)}$$
* Batch-içi negatifler: diğer satırların pozitifleri ($|B|-1$ ücretsiz negatif).
* $\mathcal{H}_i$: her satıra eklenen **4 zor negatif**:
  (i) **leksikal tuzak**: unvan karakter n-gram'ı çok benzeyen ama farklı sektörden grade-0 ilan
  (*Baklava Ustası* ↔ *Sıva Ustası*);
  (ii) **kıdem/yaka tuzağı**: aynı sektör veya ISCO grubundan, yakası zıt ya da \|Δkıdem\| ≥ 3 olan ilan
  (*Java stajyeri* ↔ *Kıdemli mimar / direktör*);
  (iii) **BM25 negatifi**: kelime kesişimi en yüksek ama grade-0 olan ilan;
  (iv) **yoğun madencilik**: eğitimsiz base modelin üst sıralara koyduğu grade-0 ilan
  (ANCE, Xiong vd., 2021; RocketQA, Qu vd., 2021). Tam yargı olduğu için yanlış negatif yoktur.
* **Sıcaklık** $\tau$ ($\text{scale}=1/\tau$) {0,05; 0,03; 0,02} ızgarasında, doğrulama NDCG@10'una göre seçilir.
* **GradCache** (Cached MNRL; Gao vd., 2021): Gradyanlar mini-batch'lere bölünerek hesaplanır. Böylece bellek sabit
  kalırken etkin batch büyür (A100'de 64 satır × 6 metin). Batch-içi negatiflerin kalitesi batch boyutuyla artar.
* **NO_DUPLICATES örnekleyici:** Aynı ilan metni bir batch'e iki kez girmez; bu, en bariz yanlış negatifi önler. Kalan
  batch-içi yanlış-negatif oranı notebook'ta ölçülüp raporlanır.
* **CoSENT çoklu-görev kaybı** (Su, 2022; AnglE: Li & Li, 2023): kademeli sıralamayı doğrudan öğretir:
  $$\mathcal{L}_{\text{CoSENT}}=\log\Big(1+\sum_{r(i)>r(j)}\exp\big(\lambda\,(s_j-s_i)\big)\Big)$$
  Grade-2 bir çift grade-1 bir çiftten, grade-1 de grade-0'dan yüksek skorlanmalıdır. MNRL yalnızca "pozitif mi
  değil mi" öğretir; CoSENT kademeyi öğretir.
* **Yapılandırılmış girdi şablonu:** `Pozisyon / Kıdem (metinden) / Konum / Yetkinlikler / Özet / Deneyim / Eğitim /
  Sertifikalar`. CV tarafında **altın etiket (sektör/yaka/ISCO) yoktur**. İlan tarafındaki *Sektör* ve *Kıdem*,
  ilanın **kendi** meta verisidir (işverenin girdiği alan).

### 3.5 Aşama 1b — BM25 (Robertson & Zaragoza, 2009)
$$\mathrm{BM25}(q,d)=\sum_{t\in q}\mathrm{idf}(t)\cdot\frac{f(t,d)(k_1+1)}{f(t,d)+k_1\big(1-b+b\frac{|d|}{\mathrm{avgdl}}\big)},\quad k_1=1{,}5,\ b=0{,}75$$
Türkçe sondan eklemeli bir dildir. Kök bulmak için **F5** (kelimenin ilk 5 karakteri) kırpması kullanılır;
Can vd. (2008) bunun Türkçe IR'de sözlük tabanlı köklemeye yakın başarı verdiğini göstermiştir. BM25 *Spring Boot,
CNC Fanuc, IATA, TIG* gibi nadir ve kesin terimlerde yoğun modellerin kaçırabileceği eşleşmeleri yakalar.
Vektörize implementasyonumuzun `rank_bm25.BM25Okapi` ile birebir aynı skoru ürettiği notebook'ta doğrulanır.

### 3.6 Reciprocal Rank Fusion (Cormack, Clarke & Büttcher, 2009)
$$\mathrm{RRF}(d)=\sum_{r\in\{\text{dense},\,\text{BM25}\}}\frac{1}{k+\mathrm{rank}_r(d)},\qquad k=60$$
* Skor ölçeklerini (kosinüs ∈ [−1,1], BM25 ∈ [0, ∞)) normalize etmeye gerek yoktur; yalnızca sıralar kullanılır.
* Parametresizdir, aşırı uyum riski yoktur ve Condorcet füzyonunu ve tekil öğrenilmiş sıralayıcıları geride bıraktığı
  gösterilmiştir.
* Top-50 aday Aşama 2'ye gider. **Recall@50**, ikinci aşamanın ulaşabileceği tavandır ve ayrıca raporlanır.

### 3.7 Aşama 2a — Cross-encoder (monoBERT tarzı)
$s_{CE}(q,d)=w^\top\,\mathrm{BERT}([\mathrm{CLS}]\,q\,[\mathrm{SEP}]\,d\,[\mathrm{SEP}])_{[\mathrm{CLS}]}$. CV ve ilan
belirteçleri **tam çapraz dikkatle** birbirini görür (Nogueira & Cho, 2019). Bi-encoder'ın tek vektöre sıkıştırma
darboğazı ortadan kalkar; *"baklava" ↔ "şerbet, yufka"* ile *"sıva" ↔ "alçı, mala"* bağlamları ayrışır. Maliyeti
$O(K)$ ileri geçiştir; bu yüzden yalnızca Top-50 üzerinde çalışır (retrieve-then-rerank; Lin vd., 2021).
**Temel model:** `BAAI/bge-reranker-v2-m3` (BGE-M3 tabanlı çok dilli reranker).
**Kayıp: LambdaLoss** (Wang vd., 2018), NDCG'ye bağlı liste-tabanlı bir sınırdır:
$$\mathcal{L}=-\sum_{r(i)>r(j)}\delta_{ij}\,\big|G_i-G_j\big|\log_2\sigma\big(s_i-s_j\big),\qquad G_i=\frac{2^{r(i)}-1}{\mathrm{IDCG}}$$
Burada $\delta_{ij}$, NDCG-Loss2++ ağırlıklandırmasıdır. Kademeli etiketleri (0–3) doğrudan kullanır.

### 3.8 Aşama 2b — LambdaMART (Burges, 2010)
Uydurma çarpanların yerine, sinyalleri birleştiren fonksiyonu **veriden öğreniriz**. LambdaRank'in gradyanı
$$\lambda_{ij}=\frac{-\sigma}{1+e^{\sigma(s_i-s_j)}}\,\big|\Delta\mathrm{NDCG}_{ij}\big|,\qquad \lambda_i=\sum_{j:\,r(i)>r(j)}\lambda_{ij}-\sum_{j:\,r(j)>r(i)}\lambda_{ji}$$
çiftlerin yer değiştirmesinin NDCG'yi ne kadar değiştirdiğiyle ölçeklenir. MART (gradyan artırmalı ağaçlar) bu
gradyanlara uydurulur. Uygulama: LightGBM `objective="lambdarank"` (Ke vd., 2017). Hiperparametreler **Optuna**
TPE ile (Akiba vd., 2019), sorgu bazlı GroupKFold içinde seçilir.

**Öznitelikler** (35 adet, `match_candidate_sota.FEATURE_COLUMNS`):
dense kosinüs/sıra/z-skor/üst-farkı · BM25 skor/sıra/z · RRF skor/sıra · CE logit/sıra/z/üst-farkı ·
**tahmini** sektör olasılığı $\hat P_{cv}(\text{sektör}=\text{ilan sektörü})$ · tahmini yaka olasılıkları ve uyumu ·
metinden kıdem, ilan seviyesi, (işaretli) fark · aynı şehir / remote / hibrit · beceri kesişimi (overlap, Jaccard) ·
unvan karakter n-gram benzerliği · deneyim yılı …

**Sızıntıya karşı üç kural:**
1. Adayın hiçbir altın etiketi öznitelik değildir. Sektör ve yaka uyumu, **yalnızca train CV'leriyle** eğitilmiş
   TF-IDF + lojistik regresyon sınıflandırıcılarının **tahmin olasılıklarıdır**. Bu sınıflandırıcıların
   val doğruluğu %100 değildir ve bu notebook'ta raporlanır: öznitelik altın etiketin kopyası değil, gerçek bir tahmindir.
2. `assert_no_gold_columns` her öznitelik tablosunda çalışır (eğitimde ve çıkarımda).
3. LambdaMART **val** sorgularında eğitilir (*stacked generalization*, Wolpert, 1992). Bu sorgular Aşama 1/2a
   modellerinin hiç görmediği sorgulardır, dolayısıyla öznitelikler örneklem-dışıdır. Cold ilanlar LTR eğitiminden
   de çıkarılır.

### 3.9 Metrikler ve istatistik
* $\mathrm{DCG@}k=\sum_{i=1}^{k}\frac{2^{r_i}-1}{\log_2(i+1)}$, $\mathrm{NDCG@}k=\mathrm{DCG@}k/\mathrm{IDCG@}k$. IDCG
  tüm havuzdan hesaplanır (Järvelin & Kekäläinen, 2002; üstel kazanç Burges vd., 2005).
  **Neden NDCG@10?** Kademeli alakayı ve sıra konumunu birlikte ölçer; ilk sayfa (10 ilan) kullanıcı deneyimidir.
* $\mathrm{MRR@10}=\frac{1}{|Q|}\sum_q \frac{1}{\mathrm{rank}_q}$ (ilk alakalı ilanın sırası; alakalı = grade ≥ 2).
  **Neden?** Kullanıcının ilk uygun ilanı ne kadar erken gördüğünü ölçer; Top-1 ile birlikte "ilk izlenim"i verir.
* P@10, R@k, Top-1. Alakalı ilan sayısı 10'dan az olan sorgularda P@10 yapısal olarak 1'e ulaşamaz. Bu yüzden
  $\mathrm{P@10}_{capped}=\text{isabet}/\min(10,|R_q|)$ da raporlanır.
* **Güven aralığı:** sorgu düzeyinde bootstrap (2.000 örnek). **Anlamlılık:** iki yönlü Fisher randomizasyon testi.
  Smucker vd. (2007) IR karşılaştırmalarında bu testi önerir.

### 3.10 Sentetik zenginleştirme
Train'de hiç (ya da yalnızca 1) CV'si olmayan uzmanlıklar için, **yalnızca WARM ilanların meslek bilgisinden**
gerçekçi Türkçe CV'ler üretildi. Bu, ilanlardan sorgu üretme fikridir (InPars: Bonifacio vd., 2022; Promptagator:
Dai vd., 2023). **Gidiş-dönüş tutarlılık filtresi** uygulandı: üretilen CV, hedefi bilmeyen bağımsız bir anotatörce
kodlandı; ISCO kodu, yaka ve kıdem (±1) hedefle uyuşmayan CV atıldı. Sentetik CV'ler **yalnızca train'e** girer.
Val/test tamamen gerçek CV'lerdir.

---

## BÖLÜM 4 — Hocanın sorabileceği en zor 10 soru ve cevapları

**S1. "Ground truth'unuzu da siz ürettiniz. Bu da döngüsel değil mi?"**
Hayır. Döngüsellik, *sıralama fonksiyonunun* doğru cevabı girdi olarak görmesidir. Bizde doğru cevap (qrels),
modellerden **bağımsız** bir süreçle, meslek kodları ve yazılı kurallarla bir kez üretilir. Hiçbir model qrels'i
veya altın meslek kodlarını öznitelik olarak görmez (`assert_no_gold_columns`). Eski sistemde ise
$c_{\text{sektör}}$, $\text{rel}$ ile **aynı** değişkendi; rastgele skorların %100 Top-1 alması bunun kanıtıdır.
Bizim sistemde rastgele skorlar bu düzeye ulaşamaz. Etiketlerin geçerliliği ayrıca IAA (Cohen κ) ve
konsolidasyon turuyla ölçüldü. Etiketlerin bir "gümüş standart" (silver standard) olduğunu açıkça kabul ediyoruz.

**S2. "Etiketleri bir LLM'e attırdınız. Bu bilimsel mi?"**
LLM destekli alaka yargısı, IR literatüründe ölçülmüş ve kabul görmüş bir yöntemdir. Thomas vd. (SIGIR 2024),
LLM yargılarının insan tercihlerini yüksek doğrulukla öngördüğünü ve Bing'de üretimde kullanıldığını raporlar;
Faggioli vd. (ICTIR 2023) güçlü ve zayıf yanlarını tartışır. Biz LLM'e "bu çift alakalı mı?" diye **sormadık**.
Yalnızca **öğe düzeyinde**, standart bir ontolojiye (ISCO-08) göre kodlama yaptırdık; çift düzeyindeki dereceler
şeffaf kurallardan türedi. Bu yaklaşım denetlenebilir: kılavuz, ham kodlar, konsolidasyon eşlemeleri ve IAA
ölçümleri depoda. Bağımsız ikinci anotasyonda ISCO-08 birim grubu için κ değeri `FAZ2_VERI_RAPORU.md`'de verilmiştir.

**S3. "Neden tek aşamalı bir cross-encoder değil de iki aşama?"**
Maliyet ve doğruluk dengesi. Cross-encoder her (CV, ilan) çifti için tam bir ileri geçiş ister:
3.000 ilan × her sorgu. Bi-encoder ilan vektörlerini önceden hesaplar; sorgu başına tek kodlama ve bir FAISS araması
yeterlidir. İki aşamalı *retrieve-then-rerank* düzeni (Nogueira & Cho, 2019; Lin vd., 2021) yüksek geri çağırmalı
ucuz bir aday üretimi ile pahalı ve hassas bir yeniden sıralamayı birleştirir. Recall@50'yi raporlayarak birinci
aşamanın ikinciye ne kadar alan bıraktığını ölçüyoruz.

**S4. "BM25'i neden hâlâ kullanıyorsunuz? Yoğun model yetmez mi?"**
Yoğun modeller anlamsal genellemede güçlüdür ama nadir, kesin teknik terimlerde (*TIG 141 prosesi, Fanuc, IATA DG,
Spring WebFlux*) zayıflayabilir. BM25 tam tersidir. RRF, iki sıralamanın **tamamlayıcı** olduğu durumda her ikisinden
de iyi sonuç verir (Cormack vd., 2009). Ablation tablosundaki "Fine-tuned dense (tek)" ile "Hybrid RRF" satırları
bu katkıyı doğrudan ölçer.

**S5. "LambdaMART'a neden ihtiyaç var? Cross-encoder skoru yetmez mi?"**
Cross-encoder yalnızca metin benzerliğini görür. Kıdem farkı, lokasyon/remote uyumu ve yaka uyumu gibi yapısal
kısıtlar metinde örtük ya da zayıf temsil edilir. Eski sistem bunları **elle seçilmiş çarpanlarla** birleştiriyordu.
LambdaMART ise bu sinyallerin ağırlıklarını ve aralarındaki doğrusal olmayan etkileşimleri (ör. "mavi yakada şehir
farkı beyaz yakadakinden daha önemli") **veriden**, doğrudan NDCG'yi optimize ederek öğrenir. Hangi sinyalin ne
kadar katkı verdiği SHAP katkılarıyla açıklanır. "Hybrid + Cross-encoder" ile "+ LambdaMART" satırları arasındaki
fark ve randomizasyon testi p-değeri bu katkıyı ölçer.

**S6. "LambdaMART'ı val kümesinde eğittiniz. Val'i hem model seçimi hem eğitim için kullanmak sızıntı değil mi?"**
Test kümesine hiç dokunulmaz; test yalnızca bir kez, en sonda kullanılır. Val'in bi-encoder/cross-encoder için erken
durdurmada kullanılması, LambdaMART'ın val özniteliklerini hafifçe iyimser görmesine yol açabilir. Bunun yönü
**test performansını düşürmek** yönündedir, şişirmek yönünde değil. Bu, yığınlamanın (Wolpert, 1992) standart
yapısıdır. Alternatif olan "train sorgularında LTR eğitimi" çok daha kötüdür: ilk aşama modelleri bu sorguları
ezberlediği için öznitelikler gerçekçi olmaz. LambdaMART hiperparametreleri val içinde sorgu bazlı GroupKFold
ile seçilir.

**S7. "Test sorgularınızdaki ilanları model eğitimde gördü. Bu transdüktif bir değerlendirme değil mi?"**
IR'de standart protokol (MS MARCO, BEIR) belge havuzunu sabit tutup **sorguları** ayrıştırmaktır; biz de öyle
yaptık. Buna ek olarak ilanların %10'unu (**cold**) hiçbir eğitim sinyalinde kullanmadık: triplet, sentetik CV,
cross-encoder listeleri ve LTR eğitiminin hiçbirinde yoklar. Notebook tüm boru hattını yalnızca bu ilanlardan oluşan
alt-korpusta yeniden çalıştırır ve sonucu ayrı bir tabloda raporlar (§11.1). Böylece görülmemiş ilanlara
genelleme doğrudan ölçülür.

**S8. "Hedeflerinizi (Top-1 > %88, NDCG@10 > %92, P@10 > %85) tutturdunuz mu?"**
Notebook hedefleri gerçek sonuçlarla otomatik ve dürüstçe karşılaştırır (§11). İki metodolojik not gerekir.
(i) Bizim alaka tanımımız eskisinden çok daha **katıdır**: "aynı sektör" değil, "aynı meslek + yaka + kıdem
(+ lokasyon)". Eski metriğin %80'i ile bizim metriğin sayısı doğrudan karşılaştırılamaz. Eski modelin gerçek
performansı aynı katı metrikle Baseline 3 satırında ölçülür.
(ii) Standart P@10, alakalı ilan sayısı 10'dan az olan sorgularda yapısal olarak 1'e ulaşamaz; bu yüzden
$P@10_{capped}$ de raporlanır. Hedef bir sayıya ulaşmak için metriği gevşetmedik. Ulaşılamadıysa bu
tabloda açıkça görünür.

**S9. "Sentetik veri modelinizi yanıltmaz mı? Test sızıntısı yaratmaz mı?"**
Sentetik CV'ler (i) yalnızca train'e girer, (ii) yalnızca WARM ilanların **meslek bilgisinden** üretilir (hiçbir
val/test CV'si kullanılmaz), (iii) hedefi bilmeyen bir anotatörün kodlamasıyla gidiş-dönüş tutarlılık filtresinden
geçer (Dai vd., 2023), (iv) başlığı bir val/test CV'siyle çakışırsa atılır. Değerlendirme %100 gerçek CV'lerle
yapılır. Sentetik verinin katkısı, sentetik satırlar çıkarılarak (`anchor_source` sütunu) ayrı bir ablation ile
ölçülebilir.

**S10. "'Baklava Ustası vs Sıva Ustası' tuzağını gerçekten çözdüğünüzü nasıl kanıtlıyorsunuz?"**
Dört bağımsız kanıt sunuyoruz:
(1) Eğitimde açıkça **leksikal tuzak negatifleri** gösterildi: unvanı karakter n-gram olarak en çok benzeyen ama
farklı sektörden grade-0 ilanlar.
(2) Vitrin adayları **test** kümesinde ve eğitimde hiç görülmedi.
(3) §12'deki tablolar, her tuzak ilanın eski sistemde ve bizim sistemde **kaçıncı sırada** çıktığını yan yana verir;
aynı çift için eski kosinüs, zero-shot, bizim dense, cross-encoder ve LambdaMART skorları da yan yana listelenir.
(4) Cross-encoder'ın `[CLS]` dikkat ağırlıkları ve LambdaMART SHAP katkıları, kararın hangi belirteçlerden
(*baklava, şerbet, yufka* ↔ *sıva, alçı*) ve hangi özniteliklerden geldiğini gösterir. Bu, tek bir örnek seçimi
değildir: tüm test kümesi üzerinden sektör bazında NDCG ve Top-1 hata analizi de raporlanır.

**Ek soru — "Neden F1/accuracy değil?"**
Görev sınıflandırma değil sıralamadır. Kullanıcı 3.000 ilanın tamamını değil, ilk 10'unu görür. F1 sıra konumunu
ve kademeli alakayı görmez. NDCG@10 ve MRR@10 ise tam olarak bu kullanıcı deneyimini ölçer (Järvelin &
Kekäläinen, 2002).

---

## BÖLÜM 5 — Sınırlılıklar (dürüst beyan)

1. **Gümüş standart etiketler.** Meslek kodları LLM destekli anotasyondur. İnsan uzman anotasyonu yapılmadı;
   uyum IAA ile ölçüldü.
2. **Veri setinin doğası.** CV ve ilanların önemli kısmı sentetik ya da şablonludur (bkz. Faz 1 §B). Gerçek
   platform verisinde mutlak sayıların farklı çıkması beklenir.
3. **Kural tabanlı dereceler.** Kurallar İK pratiğini yansıtır ama tek doğru değildir (ör. kıdem toleransı ±1).
   Kurallar `scripts/05_build_dataset.py → GRADE_RULES` içinde tek yerde tanımlıdır, değiştirilip yeniden
   üretilebilir.
4. **Bu depodaki doğrulama.** Ortamda GPU ve Hugging Face erişimi yoktu. Notebook'un hatasız çalıştığı CPU duman
   testiyle kanıtlandı; gerçek model metrikleri Colab A100 çalıştırmasından gelir.

---

## BÖLÜM 6 — Yeniden üretim

```bash
python scripts/01_parse_jobs_sql.py                 # SQL -> parquet/csv
python scripts/03_forensic_audit.py                 # Faz 1 adli denetim
python scripts/02_build_annotation_items.py         # anotasyon girdileri
#   (anotasyon: annotation/GUIDELINES.md, ham çıktılar annotation/raw, iaa_raw)
python scripts/04_consolidate_annotations.py prepare
#   (konsolidasyon: annotation/consolidation_raw)
python scripts/04_consolidate_annotations.py finalize
python scripts/06_synthetic_cvs.py targets          # sentetik CV hedefleri
#   (üretim: annotation/synthetic/gen, kör doğrulama: annotation/synthetic/verify)
python scripts/06_synthetic_cvs.py merge
python scripts/05_build_dataset.py                  # data_bundle/ + data_bundle.zip
python scripts/07_make_notebook.py                  # notebook'u üretir
python scripts/08_dry_run_notebook.py               # statik kontrol + CPU duman testi
```

---

## Kaynakça

* Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. *KDD*.
* Bonifacio, L., Abonizio, H., Fadaee, M., & Nogueira, R. (2022). InPars: Data Augmentation for Information Retrieval using Large Language Models. *SIGIR*.
* Buckley, C., & Voorhees, E. M. (2004). Retrieval Evaluation with Incomplete Information. *SIGIR*.
* Burges, C., Shaked, T., Renshaw, E., Lazier, A., Deeds, M., Hamilton, N., & Hullender, G. (2005). Learning to Rank using Gradient Descent. *ICML*.
* Burges, C. J. C. (2010). From RankNet to LambdaRank to LambdaMART: An Overview. *Microsoft Research Technical Report MSR-TR-2010-82*.
* Can, F., Kocberber, S., Balcik, E., Kaynak, C., Ocalan, H. C., & Vursavas, O. M. (2008). Information Retrieval on Turkish Texts. *JASIST*, 59(3), 407–421.
* Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., & Liu, Z. (2024). M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation (BGE-M3). *Findings of ACL*. arXiv:2402.03216.
* Cohen, J. (1960). A Coefficient of Agreement for Nominal Scales. *Educational and Psychological Measurement*, 20(1).
* Cohen, J. (1968). Weighted Kappa. *Psychological Bulletin*, 70(4).
* Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. *SIGIR*.
* Dai, Z., Zhao, V. Y., Ma, J., Luan, Y., Ni, J., Lu, J., Bakalov, A., Guu, K., Hall, K. B., & Chang, M.-W. (2023). Promptagator: Few-shot Dense Retrieval From 8 Examples. *ICLR*.
* Decorte, J.-J., Van Hautte, J., Demeester, T., & Develder, C. (2021). JobBERT: Understanding Job Titles through Skills. *FEAST @ ECML-PKDD*.
* Faggioli, G., Dietz, L., Clarke, C. L. A., Demartini, G., Hagen, M., Hauff, C., Kando, N., Kanoulas, E., Potthast, M., Stein, B., & Wachsmuth, H. (2023). Perspectives on Large Language Models for Relevance Judgment. *ICTIR*.
* Gao, L., Zhang, Y., Han, J., & Callan, J. (2021). Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup (GradCache). *RepL4NLP @ ACL*.
* Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020). Shortcut Learning in Deep Neural Networks. *Nature Machine Intelligence*, 2, 665–673.
* Henderson, M., Al-Rfou, R., Strope, B., Sung, Y., Lukács, L., Guo, R., Kumar, S., Miklos, B., & Kurzweil, R. (2017). Efficient Natural Language Response Suggestion for Smart Reply. arXiv:1705.00652.
* ILO (2012). *International Standard Classification of Occupations: ISCO-08*. International Labour Office, Geneva.
* Järvelin, K., & Kekäläinen, J. (2002). Cumulated Gain-Based Evaluation of IR Techniques. *ACM TOIS*, 20(4), 422–446.
* Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *EMNLP*.
* Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in Data Mining: Formulation, Detection, and Avoidance. *ACM TKDD*, 6(4).
* Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS*.
* Li, X., & Li, J. (2023). AnglE-optimized Text Embeddings. arXiv:2309.12871.
* Lin, J., Nogueira, R., & Yates, A. (2021). *Pretrained Transformers for Text Ranking: BERT and Beyond*. Morgan & Claypool.
* Liu, T.-Y. (2009). Learning to Rank for Information Retrieval. *Foundations and Trends in IR*, 3(3), 225–331.
* Nogueira, R., & Cho, K. (2019). Passage Re-ranking with BERT. arXiv:1901.04085.
* Qu, Y., Ding, Y., Liu, J., Liu, K., Ren, R., Zhao, W. X., Dong, D., Wu, H., & Wang, H. (2021). RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering. *NAACL*.
* Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP-IJCNLP*.
* Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in IR*, 3(4), 333–389.
* Smucker, M. D., Allan, J., & Carterette, B. (2007). A Comparison of Statistical Significance Tests for Information Retrieval Evaluation. *CIKM*.
* Su, J. (2022). CoSENT: A more efficient sentence vector scheme than Sentence-BERT. (Teknik not, kexue.fm/archives/8847.)
* Thomas, P., Spielman, S., Craswell, N., & Mitra, B. (2024). Large Language Models Can Accurately Predict Searcher Preferences. *SIGIR*.
* van den Oord, A., Li, Y., & Vinyals, O. (2018). Representation Learning with Contrastive Predictive Coding (InfoNCE). arXiv:1807.03748.
* Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R., & Wei, F. (2024). Multilingual E5 Text Embeddings: A Technical Report. arXiv:2402.05672.
* Wang, X., Li, C., Golbandi, N., Bendersky, M., & Najork, M. (2018). The LambdaLoss Framework for Ranking Metric Optimization. *CIKM*.
* Wolpert, D. H. (1992). Stacked Generalization. *Neural Networks*, 5(2), 241–259.
* Xiao, S., Liu, Z., Zhang, P., & Muennighoff, N. (2023). C-Pack: Packaged Resources to Advance General Chinese Embedding (BGE). arXiv:2309.07597.
* Xiong, L., Xiong, C., Li, Y., Tang, K.-F., Liu, J., Bennett, P., Ahmed, J., & Overwijk, A. (2021). Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval (ANCE). *ICLR*.
