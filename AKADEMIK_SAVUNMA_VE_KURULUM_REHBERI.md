# CV ↔ İş İlanı Eşleştirme — Akademik Savunma ve Kurulum Rehberi

> Bu doküman altı bölümden oluşur: (1) Colab A100 kurulumu ve çalıştırma, (2) eski sistemin bilimsel eleştirisi,
> (3) yeni mimarideki her bileşenin teorik ve matematiksel gerekçesi, (4) jüriden gelebilecek en zor 10 soru ve
> cevapları, (5) sınırlılıklar, (6) yeniden üretim. Rehberdeki her ölçüm ve istatistik, depodaki scriptlerle ve kayıtlı ara çıktılarla yeniden üretilebilir
> (BÖLÜM 6). İki istisna vardır: BÖLÜM 1'deki süreler A100'de **ölçülmemiş tahminlerdir**; nöral modellerin
> (Aşama 1/2) sonuçları ise henüz yoktur ve notebook'un Colab A100'de çalıştırılmasıyla üretilecektir. Kaynakça en sondadır.

**Teslim edilen dosyalar**

| Dosya | Ne işe yarar |
|---|---|
| `CV_Ilan_SOTA_Egitim_ve_Test_A100.ipynb` | Colab A100'de uçtan uca eğitim, test, ablation, hata analizi ve dışa aktarım (`scripts/07` üretir) |
| `data_bundle.zip` (açılmış hâli `data_bundle/`) | Sızıntısız, temiz veri paketi (Colab'a tek dosya olarak yüklenir) |
| `match_candidate_sota.py` | Eğitim ve çıkarımın ortak çekirdeği + CLI/API (`--serve` ile FastAPI) |
| `reports/FAZ1_ADLI_DENETIM_RAPORU.md` | Eski sistemin sayısal adli denetimi |
| `reports/FAZ2_VERI_RAPORU.md` | Yeni veri setinin, etiketlerin ve sızıntı kontrollerinin raporu (`scripts/10` üretir) |
| `reports/*.json`, `reports/figures/` | Ham ölçümler: `faz1_audit`, `faz2_veri_ozeti`, `annotation_iaa`, `annotation_stats`, `synthetic_cv_stats`, `qrels_validation`, `local_bm25_test`, `local_lexical_ltr`, `notebook_dogrulama`; Faz 1 grafikleri |
| `reports/dry_run_executed_notebook.ipynb` | Notebook'un CPU duman testindeki çalıştırılmış kopyası (çıktıları anlamsız, yalnız akış kanıtı) |
| `annotation/` | Kılavuz (`GUIDELINES.md`), anotasyon girdileri (`items.jsonl`, `chunks/`, `iaa_chunks/`), kayıtlı LLM çıktıları (`raw/`, `iaa_raw/`, `consolidation_raw/`, `synthetic/gen`, `synthetic/verify`, `qrels_validation/judged_*.json`), nihai etiketler (`occupation_labels.parquet`), çift düzeyi doğrulama anahtarı ve rubriği, biçim doğrulayıcıları (`validate_*.py`) |
| `scripts/01…11_*.py` (+ `common.py`) | Verinin, notebook'un ve `reports/` altındaki ölçüm çıktılarının (JSON'lar, Faz 1 grafikleri, `FAZ2_VERI_RAPORU.md`, duman testi kopyası) tamamını yeniden üreten scriptler (sıra: BÖLÜM 6); `FAZ1_ADLI_DENETIM_RAPORU.md` ise elle yazılmıştır (sayı kaynakları: BÖLÜM 6 sonu) |
| `veriseti/`, `match_candidate.py`, `test.py`, `sbertmodel.py`, `MODEL_VE_TEST_REHBERI.md` | Eski projenin verisi, kodu ve rehberi (Faz 1'de denetlenen girdiler) |

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
5. **Çalıştırın.** *Çalışma zamanı → Tümünü çalıştır*. İlk hücre kütüphaneleri sabit sürümlerle kurar (tahminen ~2 dk).
   Modeller (BGE-M3 ≈ 2,3 GB, bge-reranker-v2-m3 ≈ 2,3 GB, e5-small ≈ 0,5 GB) Hugging Face'ten otomatik iner.
6. **Bekleyin.** A100-40GB'da toplam süre **tahminen ≈ 1,5–2,5 saattir** (notebook başlığındaki tahminle aynı). Bu
   depoda GPU olmadığından süreler **ölçülmedi**; aşağıdaki kalemlerin toplamı ≈ 80–120 dk'dır, kalan pay model
   indirme, kodlama ve değerlendirme adımlarındaki değişkenlik içindir. Google Drive bağlama izni bir kez sorulur;
   sonuçlar `MyDrive/cv_ilan_sota/` klasörüne de yedeklenir:

   | Bölüm | Tahmini süre (ölçülmedi) |
   |---|---|
   | Kurulum + veri + model indirme | 5 dk |
   | Baseline 1–2 (BM25, zero-shot) | 2 dk |
   | Baseline 3 (eski yaklaşımın yeniden eğitimi, 3 epoch) | 10–15 dk |
   | Sıcaklık taraması (τ ∈ {0,05; 0,02}, 2 × yarım epoch) | 10–15 dk |
   | Aşama 1 bi-encoder (3 epoch, erken durdurma) | 25–45 dk |
   | Aşama 2a cross-encoder (2 epoch) | 15–25 dk |
   | Aşama 2b LambdaMART (Optuna 40 deneme) | 3–5 dk |
   | Ablation, soğuk-başlangıç, hata analizi, dışa aktarım | 10 dk |

7. **Sonuçları alın.** Son hücreler `export_model_bundle.zip`'i üretir (model ağırlıkları fp16; tahminen ≈2,5 GB) ve
   Drive'a kopyalar; paket 1,5 GB'tan küçükse tarayıcı indirmesini de başlatır. Grafikler, CSV tabloları ve
   `metrics.json` paketin içindedir (`figures/`, `ablation_test.csv`, `cold_start_test.csv`). §13'ün son hücresi
   (notebook'un sondan bir önceki kod hücresi) paketi sıfırdan yükleyip notebook sonuçlarıyla **uyumunu** ölçer (30 test
   sorgusunda ortalama Top-10 örtüşme ve Top-1 aynılık oranı yazdırılır; fp16 yuvarlaması küçük farklar
   yaratabileceğinden eşitlik zorunlu tutulmaz) ve `fastapi` kuruluysa FastAPI uç noktasını test eder.
8. **Canlıda kullanım (arkadaşınız için):** Kurulum satırı, paketteki `README_INFERENCE.md`'nin yazdığı satırla
   aynıdır. `attribute_models.joblib` ve `title_sim.joblib` scikit-learn nesnelerinin pickle'ları olduğundan
   scikit-learn, eğitimdeki sürüme sabitlenir (sürüm `config.json → sklearn_version` alanındadır); `--serve` için
   `fastapi` ve `uvicorn` ayrıca kurulur.
   ```bash
   unzip export_model_bundle.zip && cd export_model_bundle
   SKL=$(python -c "import json; print(json.load(open('config.json'))['sklearn_version'])")
   pip install sentence-transformers==6.1.0 transformers==5.17.0 lightgbm==4.7.0 faiss-cpu "scikit-learn==$SKL" pandas pyarrow joblib
   python match_candidate_sota.py --bundle . --cv-id 20951                  # veri setindeki bir aday
   python match_candidate_sota.py --bundle . --cv-file aday.txt --headline "Baklava Ustası"
   pip install fastapi uvicorn && python match_candidate_sota.py --bundle . --serve   # POST /match {cv_text, headline, top_k}
   ```

**Sık karşılaşılan sorunlar**

| Belirti | Çözüm |
|---|---|
| `CUDA out of memory` | 1. bölümdeki `CFG` içinde `BI_BATCH=32`, `BI_MINI_BATCH=8`, `CE_BATCH=4` yapın |
| A100 yerine T4/L4/V100 geldi | Çalışır ama yavaştır. Notebook GPU neslini algılar: bf16/TF32 yalnız Ampere+ (A100/L4) üzerinde açılır, T4/V100'de otomatik fp16 kullanılır; batch boyutlarını düşürün |
| Drive bağlamak istemiyorum | `CFG["USE_DRIVE"]=False`; sonuçlar yalnız Colab diskinde kalır (oturum kapanınca silinir) |
| Kurulum sonrası "restart runtime" uyarısı | *Çalışma zamanı → Yeniden başlat*, ardından tekrar *Tümünü çalıştır* (kurulum hızlıca atlanır) |
| Hugging Face indirmesi yavaş | Colab ağına bağlıdır; modeller oturumda bir kez iner. Model adları `CFG` içinde değiştirilebilir, ancak koddaki alternatif `intfloat/multilingual-e5-large` (≈ 2,2 GB) BGE-M3 ile hemen hemen aynı boyuttadır ve indirmeyi hızlandırmaz (bge-reranker-v2-m3 ≈ 2,3 GB yine iner). Daha küçük bir model indirmeyi kısaltır ama sonuçları değiştirir |
| Hızlı bir deneme isteniyor | `CFG`'de `BI_EPOCHS=1`, `TUNE_TEMPERATURE=False`, `LEGACY_EPOCHS=1`, `CE_EPOCHS=1`, `LTR_TRIALS=10` yapın (yukarıdaki tablodan türetilen tahmin ≈ 40–60 dk; ölçülmedi) |

**Doğrulama:** Notebook bu depoda CPU üzerinde `DRY_RUN` modunda uçtan uca çalıştırılmıştır:
`python scripts/08_dry_run_notebook.py` → `reports/notebook_dogrulama.json` (durum OK, 111 sn; çalıştırılmış kopya
`reports/dry_run_executed_notebook.ipynb`). Bu çalışma yalnızca **CPU kod yolunun** (veri yükleme, sızıntı
koruyucuları, eğitim/değerlendirme mantığı, ablation, dışa aktarım ve paketin yeniden yüklenmesi) küçük, rastgele
başlatılmış yerel BERT'ler ve alt-örneklenmiş veriyle (60 train anchor, 24 val ve 27 test sorgusu, 256 belirteç)
hatasız çalıştığını ve tablo/dosya şemalarının tutarlı olduğunu gösterir. Colab'a özgü adımlar (pip kurulumu, Drive
bağlama, dosya yükleme/indirme), GPU'ya özgü ayarlar (bf16/fp16/TF32, `adamw_torch_fused`,
`dataloader_num_workers=2`) ve gerçek BGE-M3 / bge-reranker-v2-m3 / e5-small modelleri (SentencePiece
belirteçleyici, 512 belirteç, tam veri boyutu) bu testte **çalıştırılmadı**. Rastgele modellerin metrikleri
anlamsızdır; gerçek metrikler A100 çalıştırmasından gelir.

---

## BÖLÜM 2 — Eski projedeki metodolojik hataların bilimsel eleştirisi

Tüm sayılar `reports/FAZ1_ADLI_DENETIM_RAPORU.md` ve `scripts/03_forensic_audit.py` kaynaklıdır.

### 2.1 Döngüsel değerlendirme = hedef sızıntısı (target leakage)
Eski canlı sıralama formülü (`match_candidate.py`):
$\text{Skor} = s_{\text{ham}} \times c_{\text{yaka}} \times c_{\text{kıdem}} \times c_{\text{lok}} \times c_{\text{sektör}}$.
Burada $c_{\text{sektör}}=\mathbb{1}[\text{Cat}(cv)=\text{Ind}(ilan)] + 0{,}85\,\mathbb{1}[\ne]$ (eski rehberin metninde
0,85 yerine 0,50 geçer). Eski `test.py`'nin başarı ölçütü ise $\text{rel}=2\cdot\mathbb{1}[\text{Cat}(cv)=\text{Ind}(ilan)]$.
**Canlı sıralama fonksiyonu, doğru cevap fonksiyonunun kendisini içeriyor.** Kaufman vd. (2012) bunu *leakage*
olarak tanımlar: tahmin anında meşru olarak bulunmayacak, hedefle aynı bilgiyi taşıyan bir girdinin modele verilmesi.

**Analitik kanıt.** E5 kosinüsleri dar bir bantta toplanır, $s\in[a,b]\approx[0{,}70;0{,}90]$. Yalnız sektör çarpanı
ele alındığında (diğer çarpanlar 1 alınarak) farklı sektördeki bir ilanın skoru en fazla $c\cdot b$, aynı sektördeki
bir ilanınki en az $a$ olur (tam formülde $c_{\text{yaka}}\in\{0,1\}$ olduğundan aynı sektör ilanının skoru 0'a kadar
düşebilir; aşağıdaki simülasyon da yalnız sektör çarpanını uygular). $c\cdot b<a$ ise hiçbir farklı sektör
ilanı bir aynı sektör ilanının önüne geçemez ve aday başına en az 10 aynı sektör ilanı varsa $\text{P@10}=1$ olur
(aday başına aynı sektör ilan sayısı medyan 194, minimum 8; dolayısıyla $\text{P@10}\approx 1$). Bu koşul eski
rehberdeki $c=0{,}50$ için sağlanır ($0{,}45<0{,}70$). Yukarıdaki formüldeki $c=0{,}85$ için ise **sağlanmaz**
($0{,}765>0{,}70$): bu değerde analitik garanti yoktur ve sonuç aşağıdaki ampirik kanıta dayanır.

**Ampirik kanıt.** Hiçbir bilgi içermeyen **rastgele** skorlar (yukarıdaki bant varsayımıyla $U(0{,}70;\,0{,}90)$;
kesin değerler bu varsayıma bağlıdır) `match_candidate.py`'nin sektör çarpanıyla (0,85) çarpılıp `test.py`'nin "aynı
sektör" metriğiyle ölçüldüğünde, 450 test CV'sinde **P@10 = %99,84, Top-1 = %100** elde ediliyor (0,50 ile P@10 =
%99,91). Yani canlı sıralama bu metrikle değerlendirilirse metrik modelin değil, sızdırılan
etiketin başarısını ölçer.

**Kapsam notu.** Bu simülasyon, eski rehberde bildirilen sayıların nasıl üretildiğini **anlatmaz**. Rehberdeki
P@10 = %67,87, Top-1 = %79,78, NDCG@10 = %86,93, `test.py`'nin Top-10 döngüsünden gelir; bu döngü ham kosinüs
benzerliğiyle sıralar ve sektör çarpanı uygulamaz (`soft_rerank`'te de sektör çarpanı yoktur). Bu sayıları
döngüsellik değil, NDCG'de yerel IDCG hatası (§2.2), P@10 ve Top-1'de ise zayıf "aynı sektör" alaka tanımı (§2.7)
şişirir: aynı metrikle eğitimsiz BM25 bile Top-1 = %76,44 alır (§3.11).

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
  negatiflerin ise %81,3'ü farklı sektörden (HARD_NEG'lerin %100'ü, SOFT_NEG'lerin yalnızca %34,1'i).
  Model "etiket belirtecini eşle" kısayolunu öğrenir (Geirhos vd., 2020).
  Değerlendirme de yalnızca sektör eşleşmesine baktığından sektör düzeyindeki eşleştirme ödüllendirilir; ancak
  `test.py` ve `match_candidate.py`'nin ilan metni `Sektör:` belirtecini içermez, yani belirteç kısayolu testte
  doğrudan işleyemez ve ayrıca bir train/serve uyumsuzluğu doğar.
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
  kurallarıyla yapıldı. Ardından ISCO alt-ana gruplarına göre bir **konsolidasyon** turu yapıldı; sonuçta 298 ISCO-08
  birim grubu ve 1.630 kanonik uzmanlık çıktı. Öğelerin rastgele %10'u (n = 604) **bağımsız ikinci anotasyondan**
  geçti. Uyum Cohen'in κ katsayısıyla ölçüldü (Cohen, 1960; kıdem için ağırlıklı κ, Cohen, 1968):

  | Alan (n = 604, konsolidasyon öncesi kodlar) | κ | Birebir uyum |
  |---|---|---|
  | ISCO-08 birim grubu (4 hane) | **0,916** | %91,7 |
  | Yaka | **0,980** | %99,2 |
  | Kıdem 0–5 (ikinci dereceden ağırlıklı) | **0,959** | %92,1 (\|fark\| ≤ 1: %99,8) |
  | Uzmanlık adı (grade 3'ün anahtarı; birebir yazım) | — | %61,9 |

  κ değerleri ikinci anotasyon ile ana anotasyonun **konsolidasyon öncesi** kodları arasındadır; qrels ise
  konsolidasyon sonrası etiketlerle kurulur (ISCO kodu değişen öğe 171, uzmanlık adı değişen öğe 1.447). Uzmanlık
  adlarının birebir yazım uyumu düşük olduğu için ad farklılıkları ISCO grubu düzeyindeki konsolidasyon turunda
  kanonikleştirildi; konsolidasyon sonrası uzmanlık düzeyinde ayrı bir uyum ölçümü yoktur. Grade 3 bu kanonik
  uzmanlığa dayandığından etiket belirsizliğinin en yüksek olduğu derece budur (aşağıdaki çift düzeyi doğrulamayla
  tutarlı). Landis & Koch (1977) ölçeğinde κ > 0,80 "neredeyse mükemmel" uyumdur. Ayrıca veri setindeki CV `collar`
  alanı anotatör yakasıyla tüm 3.043 CV üzerinde κ = 0,960 (%98,2) uyumludur; bu bir anotatörler arası ölçüm değil,
  bir geçerlilik kontrolüdür. Ayrıntılar `reports/FAZ2_VERI_RAPORU.md` içinde.
* **Çift düzeyinde bağımsız doğrulama:** Kurallardan türetilen dereceler, bağımsız LLM yargıçlarının doğrudan
  (CV, ilan) çiftlerine verdiği 0–3 derecelerle karşılaştırıldı. Yargıçlar kural derecesini ve altın kodları (ISCO,
  uzmanlık, yaka, kıdem) **görmedi**, yalnızca CV ve ilan metnini okudu; ancak derece tanımları kendilerine rubrik
  olarak verildi (`annotation/qrels_validation/RUBRIC.txt`) ve bu rubrik kuralların sözel karşılığıdır. Bu ölçüm bu
  yüzden kuralların bağımsız bir alaka tanımıyla değil, metinden uygulanmalarıyla tutarlılığını sınar. Örneklem: her
  kural derecesinden 90 çift olmak üzere 360 çift, yalnızca gerçek (sentetik olmayan) CV'ler. Örneklem, bölmenin
  önceki bir sürümündeki val/test CV'lerinden çekildi; güncel bölmede 360 çiftin 244'ü train, 69'u test, 47'si val
  CV'lerine düşer. Kural dereceleri bölmeden bağımsızdır ve 360 çiftin tümünde güncel qrels ile birebir aynıdır; bu
  yüzden doğrulama kurallar için geçerlidir, ancak artık held-out bir ölçüm değildir. Yargılar hiçbir eğitimde
  kullanılmaz. Sonuç: **ikinci dereceden ağırlıklı κ = 0,836**, Spearman ρ = 0,875, \|fark\| ≤ 1 oranı %97,8,
  ikili (grade ≥ 2) uyum %86,1 (κ = 0,722); birebir uyum ise %68,1'dir. 115 ayrışmanın 97'si (%84) **bir kademe
  aşağı**, 8'i iki kademe aşağı, 10'u yukarı yöndedir. Yargıçlar kural grade-3 çiftlerinin 47/90'ını (%52) 2, kural
  grade-2 çiftlerinin 43/90'ını (%48) 0–1 buldu. Yani kurallar özellikle üst derecelerde **belirgin biçimde
  cömerttir**. Kural grade-0 çiftlerin 87/90'ı yargıçça da 0'dır ve hiçbiri ≥ 2 bulunmadı. Ancak grade-0 çiftler
  **rastgele** (CV, ilan) çiftlerinden çekildi (BM25 sırası medyanı ≈ 1.410/3.000; hiçbiri BM25'in ilk 50'sinde
  değil); bu sonuç madenlenmiş zor negatiflerdeki (BM25/leksikal/yoğun) hata oranını **ölçmez**.
* **Kademeler:**

  | Grade | Kural |
  |---|---|
  | 3 | aynı uzmanlık ∧ aynı yaka ∧ \|Δkıdem\| ≤ 1 ∧ (aynı şehir ∨ Remote) |
  | 2 | aynı ISCO-08 birim grubu ∧ aynı yaka ∧ \|Δkıdem\| ≤ 2 (kıdem/lokasyonda tolere edilebilir fark) |
  | 1 | aynı yaka ∧ (aynı ana sektör ∨ aynı ISCO grubu ama \|Δkıdem\| ≥ 3) |
  | 0 | diğer |

* **Tam yargı (complete judgments):** Kademeler kurallardan türetildiği için **tüm** 3.043 × 3.000 çift etiketlidir.
  TREC tarzı *pooling*'in yarattığı "yargılanmamış = alakasız" yanlılığı (Buckley & Voorhees, 2004) burada yoktur.
  Zor negatif madenciliğinde negatifin kural derecesi bilinir, tahmin edilmez; bu yüzden **qrels'e göre** yanlış
  negatif yoktur (leksikal, BM25 ve yoğun negatifler grade 0, kıdem/yaka tuzakları grade ≤ 1). Gerçek alakaya göre
  ise yanlış negatif **imkânsız değildir** ve oranı ölçülmedi: madencilik bilerek en benzer grade-0 ilanları seçtiği için
  kural hataları tam bu slotlarda yoğunlaşabilir. Train triplet'lerinde `neg_lexical`'in 291/4.883'ü (%6,0) ve
  `neg_bm25`'in 279/4.883'ü (%5,7) anchor ile aynı ISCO-08 alt grubunda (3 hane), aynı yakada ve \|Δkıdem\| ≤ 1'dir;
  bunlar yalnızca 4 haneli birim grubu ve ana sektör farklı olduğu için grade 0 almıştır ve bir kısmı muhtemelen
  yakın-alakalıdır. Madenlenmiş negatifler üzerinde bağımsız bir yanlış-negatif ölçümü yapılmadı.

### 3.3 Sızıntısız bölme
* Sorgular (CV'ler) **ayrık** üç kümeye bölünür (70/15/15). Bölme `StratifiedGroupKFold` ile sektöre göre
  tabakalıdır. **Grup** = aynı normalize unvan ∪ neredeyse-kopya bileşeni (unvan maskeli TF-IDF cos ≥ 0,8).
  Bu yüzden test CV'lerinin hiçbirinin unvanı train'de bulunmaz.
* Vitrin adayları (Baklava ustası 20951, uçuş dispeçeri 10587, Java geliştiricisi 10511, Java stajyeri 21029)
  test kümesine sabitlenmiştir. Hata analizinde gösterilen her örnek eğitimde hiç görülmemiştir.
* İlanların %10'u (sektöre göre tabakalı) **cold** havuzudur: triplet, sentetik CV, cross-encoder listeleri ve
  LambdaMART eğitimi dahil hiçbir eğitim sinyalinde kullanılmaz; model seçimi (τ, erken durdurma, Optuna) de yalnız
  WARM ilanlarla yapılır. (Etiketsiz korpus istatistikleri, yani BM25 IDF'si ve unvan TF-IDF sözlüğü, tüm ilan
  havuzundan hesaplanır.) Ayrıca ayrı bir **soğuk-başlangıç** değerlendirmesi yapılır.
* Sentetik CV'ler yalnızca train'e girer. Başlığı bir val/test CV'siyle çakışan sentetik CV atılır.

### 3.4 Aşama 1a — Yoğun bi-encoder (DPR / Sentence-BERT)
Sorgu ve belge ayrı ayrı kodlanır: $s(q,d)=\cos(E(q),E(d))$ (Reimers & Gurevych, 2019; Karpukhin vd., 2020).
Belgeler önceden kodlanır (`job_embeddings.npy`). Çıkarımda (`SOTAMatcher.match`) sorgu bir kez kodlanır ve
normalize edilmiş 3.000 ilan vektörünün tamamı tek bir matris-vektör iç çarpımıyla puanlanır; bu, kesin (kaba
kuvvet) bir en yakın komşu aramasıdır ve FAISS `IndexFlatIP` ile eşdeğerdir. RRF ve LambdaMART'ın sıra
öznitelikleri tam-korpus sıralarını gerektirdiğinden yaklaşık indeks kullanılmaz. Pakete yazılan `faiss.index`
yalnızca daha büyük ilan havuzları için bırakılmıştır; mevcut çıkarım yolunda **sorgulanmaz**.

**Temel model: BGE-M3** (Chen vd., 2024; BGE ailesi: Xiao vd., 2023). XLM-RoBERTa-large tabanlıdır, 100+ dil
(Türkçe dahil) destekler ve 8.192 belirteçlik bağlam alır (notebook'ta girdiler 512 belirteçle kesilir). Çok dilli
MIRACL/MKQA kıyaslamalarında güçlüdür. Alternatif olarak `CFG` içinden `multilingual-e5-large` seçilebilir
(Wang vd., 2024).

**Kayıp: InfoNCE / Multiple Negatives Ranking Loss** (van den Oord vd., 2018; Henderson vd., 2017):
$$\mathcal{L}_{\text{MNRL}}=-\frac{1}{|B|}\sum_{i\in B}\log\frac{\exp(s(q_i,d_i^+)/\tau)}{\sum_{d\in\{d_j^+\}_{j\in B}\,\cup\,\bigcup_{j\in B}\mathcal{H}_j}\exp(s(q_i,d)/\tau)}$$
* Batch-içi negatifler: diğer satırların pozitifleri **ve zor negatifleri** ($|B|-1$ pozitif + $4(|B|-1)$ zor
  negatif). Sentence-Transformers her sorguyu batch'teki tüm $5|B|$ belgeyle skorlar ($|B|=64$'te 320 aday).
* $\mathcal{H}_i$: her satıra eklenen **4 zor negatif**:
  (i) **leksikal tuzak**: unvan karakter n-gram'ı çok benzeyen ama farklı sektörden grade-0 ilan
  (*Baklava Ustası* ↔ *Sıva Ustası*);
  (ii) **kıdem/yaka tuzağı**: aynı sektörden veya aynı ISCO-08 alt-ana grubundan (2 hane), yakası zıt ya da
  \|Δkıdem\| ≥ 3 olan grade ≤ 1 ilan (*Java stajyeri* ↔ *Kıdemli mimar / direktör*); böyle ilan yoksa BM25 negatif
  adaylarının 4.–10.'sundan grade-0 ilan konur (train'de 19 anchor / 29 satır);
  (iii) **BM25 negatifi**: kelime kesişimi en yüksek ama grade-0 olan ilan;
  (iv) **yoğun madencilik**: eğitimsiz base modelin üst sıralara koyduğu grade-0 ilan
  (ANCE, Xiong vd., 2021; RocketQA, Qu vd., 2021). Tam yargı sayesinde bu negatiflerin kural derecesi
  kesin bilinir: qrels'e göre (i), (iii) ve (iv) grade 0, (ii) kıdem/yaka tuzakları grade ≤ 1'dir (train'de 4.883
  tuzağın 1.730'u grade 1); gerçek alakaya göre kural kaynaklı yanlış negatif ise imkânsız değildir ve oranı ölçülmedi
  (bkz. §3.2).
* **Sıcaklık** $\tau$ ($\text{scale}=1/\tau$) {0,05; 0,02} ızgarasında (`TEMP_GRID=[20, 50]`; her değer eğitim
  satırlarının yarısıyla 1 epoch), doğrulama NDCG@10'una göre seçilir.
* **GradCache** (Cached MNRL; Gao vd., 2021): Gradyanlar mini-batch'lere bölünerek hesaplanır. Böylece bellek sabit
  kalırken etkin batch büyür (A100'de 64 satır × 6 metin). Batch-içi negatiflerin kalitesi batch boyutuyla artar.
* **Tekrarsız örnekleyici (NO_DUPLICATES):** Bir batch'te anchor ve pozitif metinleri tekildir ve hiçbir satırın
  pozitifi başka bir satırın negatifi olarak yer almaz (aynı zor negatifin birden çok satırda paylaşılması zararsızdır);
  en bariz yanlış negatif bu sayede oluşmaz. ST 6.1'in örnekleyicisinde iki sessiz hata vardı ve notebook bunları
  düzeltir. (i) Örnekleyicinin `__len__` değeri (⌈N/B⌉) gerçek batch sayısından küçüktü: çakışan satırlar sonraki,
  eksik dolu batch'lere ertelendiği için gerçekte daha fazla batch üretiliyordu. Trainer epoch başına yalnızca
  `__len__` kadar batch çektiğinden fazladaki batch'lerin satırları hiç eğitilmiyordu.
  (ii) Çoklu-veri modunda epoch tohumu iletilmediği için her epoch aynı batch'ler tekrarlanıyordu.
  Kalan batch-içi yanlış-negatif oranını notebook rastgele batch'ler üzerinden **tahmin eder** ve üç parça hâlinde
  raporlar: başka satırların pozitifleri, başka satırların zor negatifleri ve satırın kendi zor negatifleri (sonuncusu,
  kendi negatiflerin kural derecesi ≤ 1 olduğundan tanım gereği 0). Tam veriyle (A100) hesaplanmış bir oran henüz
  yoktur. CPU duman testinde (60 anchor'lık alt örneklem, $|B|=16$) hücre %1,78, %0,91 ve %0,00 yazdırmıştır; bu
  yalnız akış kanıtıdır. Tanı hücresi, NO_DUPLICATES örnekleyicisinin hiç aynı batch'e koymadığı aynı anchor'a ait
  satır çiftlerini atlamadığından ilk oran (özellikle küçük alt örneklemde) olduğundan yüksek çıkar.
* **CoSENT çoklu-görev kaybı** (Su, 2022; AnglE: Li & Li, 2023): kademeli sıralamayı doğrudan öğretir:
  $$\mathcal{L}_{\text{CoSENT}}=\log\Big(1+\sum_{r(i)>r(j)}\exp\big(\lambda\,(s_j-s_i)\big)\Big)$$
  Grade-2 bir çift grade-1 bir çiftten, grade-1 de grade-0'dan yüksek skorlanmalıdır. MNRL yalnızca "pozitif mi
  değil mi" öğretir; CoSENT kademeyi öğretir.
* **Yapılandırılmış girdi şablonu:** `Pozisyon / Kıdem (metinden) / Konum / Yetkinlikler / Özet / Deneyim / Eğitim /
  Sertifikalar / Projeler / Diller` (boş bölümler yazılmaz). CV tarafında **altın etiket (sektör/yaka/ISCO) yoktur**.
  İlan tarafındaki *Sektör* ve *Kıdem*, ilanın **kendi** meta verisidir (işverenin girdiği alan).

### 3.5 Aşama 1b — BM25 (Robertson & Zaragoza, 2009)
$$\mathrm{BM25}(q,d)=\sum_{t\in q}\mathrm{idf}(t)\cdot\frac{f(t,d)(k_1+1)}{f(t,d)+k_1\big(1-b+b\frac{|d|}{\mathrm{avgdl}}\big)},\quad k_1=1{,}5,\ b=0{,}75$$
Türkçe sondan eklemeli bir dildir. Kök bulmak için **F5** (kelimenin ilk 5 karakteri) kırpması kullanılır;
Can vd. (2008) bunun Türkçe IR'de sözlük tabanlı köklemeye yakın başarı verdiğini göstermiştir. BM25 *Spring Boot,
CNC Fanuc, IATA, TIG* gibi nadir ve kesin terimlerde yoğun modellerin kaçırabileceği eşleşmeleri yakalar.
Vektörize implementasyonumuzun `rank_bm25.BM25Okapi` ile birebir aynı skoru ürettiği notebook'ta doğrulanır.

### 3.6 Reciprocal Rank Fusion (Cormack, Clarke & Büttcher, 2009)
$$\mathrm{RRF}(d)=\sum_{r\in\{\text{dense},\,\text{BM25}\}}\frac{1}{k+\mathrm{rank}_r(d)},\qquad k=60$$
* Skor ölçeklerini (kosinüs ∈ [−1,1], BM25 ∈ [0, ∞)) normalize etmeye gerek yoktur; yalnızca sıralar kullanılır.
* Ayarlanan parametresi yoktur ($k=60$ literatürdeki sabit değerdir), bu yüzden aşırı uyum riski taşımaz;
  Condorcet füzyonunu ve tekil öğrenilmiş sıralayıcıları geride bıraktığı gösterilmiştir.
* Top-50 aday Aşama 2'ye gider. **Recall@50**, ikinci aşamanın ulaşabileceği tavandır ve ayrıca raporlanır.

### 3.7 Aşama 2a — Cross-encoder (monoBERT tarzı)
$s_{CE}(q,d)=w^\top\,\mathrm{BERT}([\mathrm{CLS}]\,q\,[\mathrm{SEP}]\,d\,[\mathrm{SEP}])_{[\mathrm{CLS}]}$. CV ve ilan
belirteçleri **tam çapraz dikkatle** birbirini görür (Nogueira & Cho, 2019). Bi-encoder'ın tek vektöre sıkıştırma
darboğazı ortadan kalkar; ilke olarak *"baklava" ↔ "şerbet, yufka"* ile *"sıva" ↔ "alçı, mala"* bağlamları
ayrıştırılabilir (bunun gerçekleşip gerçekleşmediği §12'de sınanır, bkz. S10). Maliyeti
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
2. Öznitelikler yapısal olarak altın etiketsizdir: `build_pair_features` girdi olarak yalnızca
   `QueryMeta.from_text` çıktısını (CV metni + train'de eğitilmiş sınıflandırıcıların olasılıkları), `JobTable`'ı
   (ilanın kendi metni ve meta verisi + train'de eğitilmiş yaka sınıflandırıcısının ilan unvanından tahmini), unvan benzerliği
   modelini ve Aşama 1/2 skorları ile sıralarını (dense, BM25, RRF, CE; yalnızca CV ve ilan metninden hesaplanır)
   alır; bunların hiçbiri altın etiket değildir ve çıktısı daima `FEATURE_COLUMNS` ile sınırlıdır.
   `assert_no_gold_columns` bunun üstünde yalnızca bir **şema korumasıdır**: öznitelik tablosunda `gold_`/`legacy_` önekli ya da `grade`, `label`,
   `Category`, `collar`, `occ_key`, `isco4`, `specialty` adlı bir sütun varsa hata verir. Notebook'ta LTR eğitim/test
   tablolarında, soğuk-başlangıç ve hata analizi tablolarında; ayrıca `scripts/11_local_lexical_ltr.py`'de ve
   çıkarımda (`SOTAMatcher.match`) çağrılır. Etiket sütunlarının yanlışlıkla öznitelik olarak verilmesini engeller;
   değerleri denetlemediği için nötr adlı bir sütuna taşınmış altın bilgiyi yakalayamaz. Asıl güvence,
   özniteliklerin nasıl üretildiğidir (kural 1).
3. LambdaMART **val** sorgularında eğitilir (*stacked generalization*, Wolpert, 1992). Aşama 1/2a modelleri bu
   sorgularla hiç **eğitilmedi** (gradyan almadı); ancak val, Aşama 1/2a'nın **model seçiminde** kullanıldı: τ
   taraması; bi-encoder'da val NDCG@10'a göre erken durdurma ve en iyi checkpoint seçimi (val triplet'lerinden val
   kaybı da izlenir); cross-encoder'da aynı biçimde erken durdurma ve checkpoint seçimi. Bu yüzden val öznitelikleri eğitim açısından
   örneklem-dışıdır, ama test özniteliklerine göre en fazla hafifçe iyimserdir (bkz. S6). Cold ilanlar LTR
   eğitiminden de çıkarılır.

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
Hedeflerin seçildiği andaki (önceki) bölmede train'de hiç (ya da yalnızca 1) gerçek CV'si olmayan 868 WARM uzmanlık
için (nihai bölmede bu koşulu 890 uzmanlık sağlar; 868 hedefin 61'inin artık train'de ≥ 2 gerçek CV'si vardır),
**yalnızca WARM ilanların meslek bilgisinden** uzmanlık başına bir gerçekçi Türkçe CV üretildi. Bu, ilanlardan sorgu üretme fikridir (InPars: Bonifacio vd., 2022; Promptagator:
Dai vd., 2023). **Gidiş-dönüş tutarlılık filtresi** uygulandı: üretilen CV, hedefi bilmeyen bağımsız bir anotatörce
kodlandı; ISCO kodu, yaka ve kıdem (±1) hedefle uyuşmayan CV atıldı. 868 CV'nin **710'u** kabul edildi; 157'si
gidiş-dönüş filtresiyle, 1'i başlığı bir val/test CV'siyle çakıştığı için reddedildi. Sentetik CV'ler **yalnızca
train'e** girer (train: 2.125 gerçek + 710 sentetik = 2.835 CV). Val (457) ve test (461) tamamen gerçek CV'lerdir.

### 3.11 Bu depoda ölçülen gerçek sonuçlar (GPU gerektirmeyen bileşenler)
Ortamda GPU ve Hugging Face erişimi olmadığından nöral modellerin sayıları Colab çalıştırmasından gelecektir. Nöral
olmayan bileşenler ise yeni sızıntısız test kümesinde (453 sorgu; alakalı ilanı olmayan 8 sorgu TREC geleneğiyle
dışarıda) **gerçekten** ölçüldü (`reports/local_bm25_test.json`, `reports/local_lexical_ltr.json`):

| Sistem (test, kademeli qrels) | NDCG@10 [%95 GA] | MRR@10 | P@10 | P@10_capped | R@10 | Top-1 |
|---|---|---|---|---|---|---|
| BM25 (Türkçe F5) | 53,66 [51,5; 55,8] | 74,03 | 38,37 | 48,53 | 26,59 | 64,68 |
| BM25 Top-50 + LambdaMART (yalnız metin-türevli öznitelikler; dense/CE yok) | **66,05** [63,9; 68,2] | **80,23** | **45,01** | **56,35** | **29,71** | **72,85** |

NDCG@10 farkı Fisher randomizasyon testinde anlamlıdır (p ≈ 0,0001: 10.000 permütasyonun hiçbiri gözlenen farka
ulaşmadı, bu testin ulaşabildiği en küçük değerdir). Ham (çarpansız) BM25 sıralamasının üstüne eklenen **öğrenilen
sıralama**, nöral bileşenler olmadan bile NDCG@10'u 12,4 puan (53,66 → 66,05), Top-1'i 8,2 puan (64,68 → 72,85)
artırıyor. Bu karşılaştırma eski elle seçilmiş çarpanlara karşı değil, çarpan içermeyen BM25'e karşıdır. Eski
yaklaşımın (e5-small + `CosineSimilarityLoss`) sızıntısız ölçümü notebook'taki Baseline 3 satırındadır; eski
çarpanlar altın etiket kullandığından sızıntısız karşılaştırmaya giremez (notebook §6.1'de yalnız sızıntı gösterimi
olarak yer alır). Notebook'taki tam boru hattı bu iki sistemin üstüne ince ayarlı dense getirim ve cross-encoder ekler;
onların sayıları henüz yoktur.

**Aynı BM25 sıralamasının eski metrikle ölçümü:** "aynı sektör" ölçütüyle P@10 = %64,12, Top-1 = %78,09. Eski metrik
aynı sıralamayı kademeli metriğe göre P@10'da 25,8 puan (38,37 → 64,12; P@10_capped'e göre 15,6 puan), Top-1'de
13,4 puan (64,68 → 78,09) daha yüksek gösteriyor (eski metrik 461 test CV'sinin tamamında, kademeli metrikler alakalı
ilanı olan 453 sorguda ortalanır). Faz 1'de eski 450 test CV'si üzerinde de aynı karşılaştırma yapıldı: eski metrikle
eğitimsiz BM25 Top-1 = %76,44 alıyor. Eski projenin ince ayarlı model için bildirdiği (çarpansız, ham benzerlikle
üretilmiş; §2.1) %79,78, bundan yalnızca 3,3 puan yüksek.

---

## BÖLÜM 4 — Hocanın sorabileceği en zor 10 soru ve cevapları

**S1. "Ground truth'unuzu da siz ürettiniz. Bu da döngüsel değil mi?"**
Hayır. Döngüsellik, *sıralama fonksiyonunun* doğru cevabı girdi olarak görmesidir. Bizde doğru cevap (qrels),
modellerden **bağımsız** bir süreçle, meslek kodları ve yazılı kurallarla bir kez üretilir. Hiçbir model qrels'i
veya altın meslek kodlarını öznitelik olarak görmez: öznitelikler yalnızca CV metninden ve ilanın kendi metni
ile meta verisinden üretilir (§3.8 kural 1–2; `assert_no_gold_columns` bunu sütun adı düzeyinde ayrıca denetler).
Eski canlı sıralamada (`match_candidate.py`) ise $c_{\text{sektör}}$, $\text{rel}$ ile **aynı** alandan (adayın altın
`Category`'si) türetiliyordu; rastgele skorlar yalnızca bu sektör çarpanıyla (0,85) çarpılıp aynı metrikle
ölçüldüğünde bile %100 Top-1 alır (§2.1).
Eski rehberdeki sayılar ise çarpansız üretildi; onların sorunu döngüsellik değil, yerel IDCG ve zayıf "aynı sektör"
alaka tanımıdır. Bizim sistemde rastgele skorlar bu düzeye ulaşamaz. Etiketlerin geçerliliği ayrıca IAA (Cohen κ)
ve çift düzeyi doğrulamayla ölçüldü. Etiketlerin bir "gümüş standart" (silver standard) olduğunu açıkça kabul ediyoruz.

**S2. "Etiketleri bir LLM'e attırdınız. Bu bilimsel mi?"**
LLM destekli alaka yargısı, IR literatüründe ölçülmüş ve kabul görmüş bir yöntemdir. Thomas vd. (SIGIR 2024),
LLM yargılarının insan tercihlerini yüksek doğrulukla öngördüğünü ve Bing'de üretimde kullanıldığını raporlar;
Faggioli vd. (ICTIR 2023) güçlü ve zayıf yanlarını tartışır. qrels'i üretmek için LLM'e "bu çift alakalı mı?" diye
**sormadık**. Yalnızca **öğe düzeyinde**, standart bir ontolojiye (ISCO-08) göre kodlama yaptırdık; çift düzeyindeki
dereceler şeffaf kurallardan türedi. Çift düzeyindeki LLM yargıları yalnızca doğrulama için (360 çift) kullanıldı ve
hiçbir eğitime girmedi. Bu yaklaşım denetlenebilir: kılavuz, ham kodlar, konsolidasyon eşlemeleri ve IAA ölçümleri
depoda. Bağımsız ikinci anotasyonda (konsolidasyon öncesi kodlar) ISCO-08 birim grubu için **κ = 0,916**, yaka için
**κ = 0,980** çıktı; uzmanlık adlarında birebir yazım uyumu ise yalnızca %61,9'dur. Kural-türevi dereceler, kural
derecesini görmeyen (derece tanımlarını rubrik olarak alan) yargıçların çift düzeyindeki yargılarıyla **ağırlıklı
κ = 0,836** uyumlu; birebir uyum %68,1'dir ve ayrışmalar çoğunlukla kuralın bir kademe cömert olduğu yöndedir (§3.2).

**S3. "Neden tek aşamalı bir cross-encoder değil de iki aşama?"**
Maliyet ve doğruluk dengesi. Cross-encoder her (CV, ilan) çifti için tam bir ileri geçiş ister:
3.000 ilan × her sorgu. Bi-encoder ilan vektörlerini önceden hesaplar; sorgu başına tek kodlama ve 3.000 ilan
vektörüyle tek bir iç çarpım (FAISS `IndexFlatIP` ile eşdeğer, kesin kaba kuvvet arama) yeterlidir. İki aşamalı
*retrieve-then-rerank* düzeni (Nogueira & Cho, 2019; Lin vd., 2021) yüksek geri çağırmalı ucuz bir aday üretimi ile
pahalı ve hassas bir yeniden sıralamayı birleştirir. Recall@50'yi raporlayarak birinci
aşamanın ikinciye ne kadar alan bıraktığını ölçüyoruz.

**S4. "BM25'i neden hâlâ kullanıyorsunuz? Yoğun model yetmez mi?"**
Yoğun modeller anlamsal genellemede güçlüdür ama nadir, kesin teknik terimlerde (*TIG 141 prosesi, Fanuc, IATA DG,
Spring WebFlux*) zayıflayabilir. BM25 tam tersidir. RRF, iki sıralamanın **tamamlayıcı** olduğu durumda her ikisinden
de iyi sonuç verir (Cormack vd., 2009). Notebook §11'deki "Bileşen katkıları" tablosunun (`ablation_component_tests.csv`)
"BM25 füzyonu (S1 vs fine-tuned dense)" satırı bu katkıyı ΔNDCG@10, ΔTop-1 ve eşleştirilmiş randomizasyon
p-değerleriyle doğrudan ölçer. Bu değerler A100 çalıştırmasından gelecektir; bu depoda henüz yoktur.

**S5. "LambdaMART'a neden ihtiyaç var? Cross-encoder skoru yetmez mi?"**
Cross-encoder yalnızca metin benzerliğini görür. Kıdem farkı, lokasyon/remote uyumu ve yaka uyumu gibi yapısal
kısıtlar metinde örtük ya da zayıf temsil edilir. Eski sistem bunları **elle seçilmiş çarpanlarla** birleştiriyordu.
LambdaMART ise bu sinyallerin ağırlıklarını ve aralarındaki doğrusal olmayan etkileşimleri (ör. kıdem farkının ve
lokasyon uyumunun ancak meslek ve yaka uyuştuğunda dereceyi değiştirmesi) **veriden**, doğrudan NDCG'yi optimize
ederek öğrenir. Hangi sinyalin ne kadar katkı verdiği SHAP katkılarıyla açıklanır. Notebook §11'deki "Bileşen katkıları" tablosu
(`ablation_component_tests.csv`) her bileşeni ayrı ayrı, aynı test sorgularında ΔNDCG@10, ΔTop-1 ve iki yönlü
eşleştirilmiş randomizasyon p-değerleriyle test eder: ince ayar (fine-tuned dense vs zero-shot base), BM25 füzyonu
(S1 vs fine-tuned dense), cross-encoder (Hybrid+CE vs S1), **LambdaMART (S2 vs Hybrid+CE)** ve toplam (S2 vs B1 ile
S2 vs B3). LambdaMART'ın cross-encoder üzerindeki katkısı bu tablonun "LambdaMART (S2 vs Hybrid+CE)" satırıdır; ana
ablation tablosundaki "önceki satıra göre" p-değeri ise S2'yi S1 ile karşılaştırdığından cross-encoder ve LambdaMART'ı
birlikte ölçer. Bu tablonun değerleri A100 çalıştırmasından gelecektir; bu depoda henüz yoktur. Nöral bileşenler
olmadan yerelde ölçülen karşılaştırma ise LambdaMART'ın ham BM25 üzerindeki katkısıdır (cross-encoder üzerindeki
değil): BM25 → BM25 + LambdaMART geçişi NDCG@10'u %53,66'dan %66,05'e (NDCG@10 farkı için p ≈ 0,0001; §3.11),
Top-1'i %64,68'den %72,85'e çıkarıyor.

**S6. "LambdaMART'ı val kümesinde eğittiniz. Val'i hem model seçimi hem eğitim için kullanmak sızıntı değil mi?"**
Test kümesine hiç dokunulmaz; test yalnızca en sonda ölçüm için kullanılır ve hiçbir seçim test üzerinde yapılmaz.
Val'in bi-encoder/cross-encoder için τ seçimi, erken durdurma ve en iyi checkpoint seçiminde kullanılması (gradyan
eğitiminde değil), LambdaMART'ın val özniteliklerini hafifçe iyimser görmesine yol açabilir. Bunun yönü
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
Bunlar **hedeftir**, ulaşılmış sonuç değildir: nöral sistemlerin sayıları A100 çalıştırmasından gelecektir ve bu depoda
henüz yoktur. Notebook hedefleri gerçek sonuçlarla otomatik ve dürüstçe karşılaştırır (§11; "KARŞILANDI /
karşılanmadı"). Bu depoda yerelde ölçülen en iyi sistem, nöral bileşen içermeyen BM25 + LambdaMART'tır: NDCG@10 =
%66,05, Top-1 = %72,85, P@10 = %45,01 (P@10_capped = %56,35). İki metodolojik not gerekir.
(i) Bizim alaka tanımımız eskisinden çok daha **katıdır**: "aynı sektör" değil, "aynı meslek + yaka + kıdem
(+ lokasyon)". Eski metriğin %80'i ile bizim metriğin sayısı doğrudan karşılaştırılamaz. Eski yaklaşımın
(e5-small + `CosineSimilarityLoss`) aynı katı metrikle sızıntısız ölçümü Baseline 3 satırındadır; bu satır eski modelin
kendisi değil, eski çiftlerin yalnız yeni train CV'lerine ve WARM ilanlara düşen kısmıyla (16.206 / 25.906 çift)
yeniden eğitilmiş kopyasıdır ve sayısı A100 çalıştırmasından gelecektir.
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
(3) §12'deki tuzak tablosu, her vitrin adayı için unvanı karakter n-gram olarak en çok benzeyen beş grade-0 ilanın
eski yaklaşımda (B3) ve bizim sistemde **kaçıncı sırada** çıktığını yan yana verir. Unvan benzerliği modelden bağımsız
ve deterministik olduğundan bu liste varsayılan `CFG` ile her çalıştırmada aynıdır (duman testi çıktısında da
böyledir); 20951 için *Gemi Montaj Ustası, Zemin Şap Ustası, Kalıp ve Demir Ustası, Kaba ve İnce Sıva Ustası, Isı ve
Ses Yalıtım Ustası*. §12.1 ise doğru ilan ile unvanı en çok benzeyen grade-0 tuzak ilan çifti için (20951'de bu
*Gemi Montaj Ustası*'dır, Sıva ilanı değil) eski kosinüs, zero-shot, bizim dense, cross-encoder ve LambdaMART
skorlarını yan yana listeler.
(4) Aynı çift için notebook, cross-encoder'ın son katmanında başlıklar üzerinden ortalanmış `[CLS]` → ilan
belirteçleri dikkat ağırlıklarını alt-belirteçlerden kelimelere toplar ve en yüksek 12 kelimeyi çizer; LambdaMART
SHAP katkıları da kararı hangi özniteliklerin belirlediğini gösterir. Bakılacak şey, doğru ilanda meslek içeriği
taşıyan kelimelerin (ör. *baklava, şerbet, yufka*) öne çıkıp çıkmadığı ve tuzak ilanın CE ve LambdaMART skorunun
doğru ilanın altında kalıp kalmadığıdır. Bu çıktılar A100 çalıştırmasından gelir (duman testindeki rastgele
modellerin dikkatleri anlamsızdır) ve dikkat ağırlıkları nedensel bir açıklama değil, tamamlayıcı bir göstergedir.
Bu, tek bir örnek seçimi değildir: tüm test kümesi üzerinden sektör bazında NDCG ve Top-1 hata analizi de raporlanır.

**Ek soru — "Neden F1/accuracy değil?"**
Görev sınıflandırma değil sıralamadır. Kullanıcı 3.000 ilanın tamamını değil, ilk 10'unu görür. F1 sıra konumunu
ve kademeli alakayı görmez. NDCG@10 ve MRR@10 ise tam olarak bu kullanıcı deneyimini ölçer (Järvelin &
Kekäläinen, 2002).

---

## BÖLÜM 5 — Sınırlılıklar (dürüst beyan)

1. **Gümüş standart etiketler.** Meslek kodları LLM destekli anotasyondur. İnsan uzman anotasyonu yapılmadı;
   uyum IAA ile ölçüldü (κ = 0,92 ISCO, 0,98 yaka; uzmanlık adında birebir yazım uyumu %61,9). Çift düzeyi
   doğrulama, kuralların özellikle üst derecelerde belirgin biçimde cömert olduğunu gösteriyor: yargıçlar kural
   grade-3 çiftlerinin %52'sini (47/90) 2, grade-2 çiftlerinin %48'ini (43/90) 0–1 buldu. Mutlak NDCG değerleri bu
   yüzden iyimser olabilir. Tüm sistemler aynı qrels ile ölçülür; ancak sapma sistematik olduğundan bu yanlılığın
   sistemleri eşit etkilediği kanıtlanmış değildir (ör. kural qrels'iyle eğitilen LTR kural etiketlerine daha yakın
   eşleşmeleri kayırabilir). Yargı-düzeltilmiş derecelerle yeniden ölçüm yapılmadı. Doğrulama örneklemindeki grade-0
   çiftleri rastgele olduğundan, zor negatiflerdeki kural hataları da ölçülmedi (§3.2).
2. **Teknoloji yığını ayrımı yapılmadı.** Kılavuz gereği "Java backend" ile ".NET backend" aynı uzmanlıktır (grade 3
   olabilir). Teknoloji uyumu etikette değil öznitelikte (beceri kesişimi) yer alır; jüri isterse uzmanlık tanımı
   inceltilerek qrels yeniden üretilebilir.
3. **Veri setinin doğası.** CV'lerin önemli kısmı şablon cümleler taşır (CV'lerin %38,3'ü en az bir şablon cümle
   içerir; bkz. Faz 1 §B) ve train'deki 710 CV sentetiktir (§3.10). İlanlardaki şablon tekrarı Faz 1'de ölçülmedi.
   Gerçek platform verisinde mutlak sayıların farklı çıkması beklenir.
4. **Kural tabanlı dereceler.** Kurallar İK pratiğini yansıtır ama tek doğru değildir (ör. kıdem toleransı ±1).
   Dereceler `scripts/05_build_dataset.py → grade_matrix()` içinde hesaplanır (kıdem toleransları `sd <= 1` /
   `sd <= 2`, yaka ve lokasyon koşulları). Aynı dosyadaki `GRADE_RULES` yalnızca bu kuralların `metadata.json`'a
   yazılan açıklamasıdır; değiştirilmesi hiçbir dereceyi değiştirmez. Kural değiştirilecekse `grade_matrix()`
   düzenlenip 05 yeniden çalıştırılır (qrels, triplet'ler ve paket yeniden üretilir; ardından BÖLÜM 6'daki sonraki
   adımlar). Açıklama metinleri (`GRADE_RULES`, paket README'si, `09_qrels_validation.py → RUBRIC`) elle güncellenir.
5. **Bu depodaki doğrulama.** Ortamda GPU ve Hugging Face erişimi yoktu. CPU duman testi (DRY_RUN; minik rastgele
   modeller, alt-örneklenmiş veri) notebook'un CPU kod yolunun ve çıktı şemalarının hatasız çalıştığını gösterdi;
   Colab, GPU (bf16/TF32) ve gerçek model dalları bu ortamda çalıştırılamadı. Gerçek model metrikleri ve süreler Colab
   A100 çalıştırmasından gelir.

---

## BÖLÜM 6 — Yeniden üretim

LLM'e yaptırılan adımların (meslek anotasyonu, IAA ikinci anotasyonu, konsolidasyon, sentetik CV üretimi ve kör
doğrulaması, çift düzeyi yargılar) çıktıları depoda kayıtlıdır. Aşağıdaki tarif bu kayıtlı çıktılar üzerinde yalnızca
**deterministik** adımları yeniden çalıştırır (sabit tohumlar); LLM adımlarını yeniden yapmak gerekmez. Komutlar depo
kökünden çalıştırılır.

```bash
python scripts/01_parse_jobs_sql.py                   # SQL -> work/tum_ilanlar.parquet / .csv
python scripts/02_build_annotation_items.py           # -> annotation/items.jsonl, chunks/, iaa_chunks/
#   kayıtlı LLM anotasyonu: annotation/raw/ (68 parça), annotation/iaa_raw/ (6 parça); kılavuz: annotation/GUIDELINES.md
python scripts/04_consolidate_annotations.py prepare  # -> annotation/merged_raw.parquet, reports/annotation_iaa.json,
                                                      #    annotation/consolidation_inputs/
#   kayıtlı LLM konsolidasyonu: annotation/consolidation_raw/ (43 ISCO grubu)
python scripts/04_consolidate_annotations.py finalize # -> annotation/occupation_labels.parquet, reports/annotation_stats.json
#   kayıtlı sentetik CV hedefleri, üretim ve kör doğrulama: annotation/synthetic/{specs,gen,verify}/
#   (06 ... targets YALNIZCA sentetik CV'ler yeniden üretilecekse çalıştırılır: specs/'i silip yeniden yazar.
#    Kayıtlı specs/ (868 hedef, 35 parti) önceki bir bölmeyle üretildi; mevcut bölmeyle varsayılan --per-batch 25
#    --max 900 -> 890 hedef, 36 parti verir ve kayıtlı gen/, verify/ ile eşleşmez; ardından gen/ ve verify/ yeniden
#    üretilmelidir)
python scripts/06_synthetic_cvs.py merge              # gidiş-dönüş filtresi -> annotation/synthetic/synthetic_cvs.json,
                                                      #    reports/synthetic_cv_stats.json (868 hedef, 710 kabul)
python scripts/05_build_dataset.py                    # split, qrels, triplet, sızıntı kontrolleri -> data_bundle/,
                                                      #    data_bundle.zip, reports/faz2_veri_ozeti.json
#   (06 merge bölmeyi etiketlerden kendisi hesaplar; 05'ten önce çalışır. synthetic_cvs.json değişirse 05 yeniden çalıştırılır.)
#   kayıtlı çift düzeyi LLM yargıları: annotation/qrels_validation/judged_*.json + annotation/qrels_validation_key.json
#   (09 ... sample YALNIZCA yeniden yargılanacaksa çalıştırılır: anahtarı ve pairs_*.json'u mevcut data_bundle'dan
#    yeniden çizer. Kayıtlı anahtar önceki bir bölmeden çekildiği için mevcut veriden yeniden üretilmez (kural
#    dereceleri mevcut qrels ile aynıdır, score geçerlidir); sample çalıştırılırsa kayıtlı judged_*.json geçersiz
#    olur, yeniden yargılama gerekir)
python scripts/09_qrels_validation.py score           # -> reports/qrels_validation.json
python scripts/03_forensic_audit.py                   # Faz 1 -> reports/faz1_audit.json, reports/figures/faz1_*.png
python scripts/11_local_lexical_ltr.py                # BM25 vs BM25+LambdaMART -> reports/local_lexical_ltr.json
python scripts/10_make_faz2_report.py                 # yerel BM25 -> reports/local_bm25_test.json;
                                                      #    -> reports/FAZ2_VERI_RAPORU.md (11'in çıktısını okur)
python scripts/07_make_notebook.py                    # -> CV_Ilan_SOTA_Egitim_ve_Test_A100.ipynb
python scripts/08_dry_run_notebook.py --bundle data_bundle.zip --workdir work/dry_run --kernel python3
                                                      # statik kontrol + CPU duman testi -> work/dry_run/executed.ipynb,
                                                      #    reports/notebook_dogrulama.json (--kernel: Jupyter çekirdeğinin adı)
```

`03_forensic_audit.py` yalnızca 01'in çıktısına, `veriseti/`'ye ve eski `test.py`'ye bağlıdır ve 01'den sonra
herhangi bir noktada çalıştırılabilir. Markdown raporu yazmaz; elle yazılan `reports/FAZ1_ADLI_DENETIM_RAPORU.md`
ölçüm sayılarının çoğunu `faz1_audit.json`'dan alır. Birkaç statik gözlem (eski rehberin metrikleri ve çarpanları,
eski bölme özetindeki train grup sayısı ve büyüklükleri, "Distribütör ve Bayi Yöneticisi" ilanının açıklama uzunluğu)
doğrudan kaynak dosyalardan okunmuştur; betik çıktısında yer almayan birkaç ek kontrol de metinde *(ayrıca kontrol
edildi)* diye işaretlenmiştir (raporun başındaki nota bakın). Nöral sonuçlar bu tarifle değil, notebook'un Colab A100'de çalıştırılmasıyla üretilir
(BÖLÜM 1).

---

## Kaynakça

* Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. *KDD*.
* Bonifacio, L., Abonizio, H., Fadaee, M., & Nogueira, R. (2022). InPars: Unsupervised Dataset Generation for Information Retrieval. *SIGIR* (kısa bildiri), 2387–2392. https://doi.org/10.1145/3477495.3531863
* Buckley, C., & Voorhees, E. M. (2004). Retrieval Evaluation with Incomplete Information. *SIGIR*.
* Burges, C., Shaked, T., Renshaw, E., Lazier, A., Deeds, M., Hamilton, N., & Hullender, G. (2005). Learning to Rank using Gradient Descent. *ICML*.
* Burges, C. J. C. (2010). From RankNet to LambdaRank to LambdaMART: An Overview. *Microsoft Research Technical Report MSR-TR-2010-82*.
* Can, F., Kocberber, S., Balcik, E., Kaynak, C., Ocalan, H. C., & Vursavas, O. M. (2008). Information Retrieval on Turkish Texts. *JASIST*, 59(3), 407–421.
* Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., & Liu, Z. (2024). M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation (BGE-M3). *Findings of ACL*. arXiv:2402.03216.
* Cohen, J. (1960). A Coefficient of Agreement for Nominal Scales. *Educational and Psychological Measurement*, 20(1).
* Cohen, J. (1968). Weighted Kappa. *Psychological Bulletin*, 70(4).
* Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. *SIGIR*.
* Dai, Z., Zhao, V. Y., Ma, J., Luan, Y., Ni, J., Lu, J., Bakalov, A., Guu, K., Hall, K. B., & Chang, M.-W. (2023). Promptagator: Few-shot Dense Retrieval From 8 Examples. *ICLR*.
* Faggioli, G., Dietz, L., Clarke, C. L. A., Demartini, G., Hagen, M., Hauff, C., Kando, N., Kanoulas, E., Potthast, M., Stein, B., & Wachsmuth, H. (2023). Perspectives on Large Language Models for Relevance Judgment. *ICTIR*.
* Gao, L., Zhang, Y., Han, J., & Callan, J. (2021). Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup (GradCache). *RepL4NLP @ ACL*.
* Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020). Shortcut Learning in Deep Neural Networks. *Nature Machine Intelligence*, 2, 665–673.
* Henderson, M., Al-Rfou, R., Strope, B., Sung, Y., Lukács, L., Guo, R., Kumar, S., Miklos, B., & Kurzweil, R. (2017). Efficient Natural Language Response Suggestion for Smart Reply. arXiv:1705.00652.
* ILO (2012). *International Standard Classification of Occupations: ISCO-08*. International Labour Office, Geneva.
* Järvelin, K., & Kekäläinen, J. (2002). Cumulated Gain-Based Evaluation of IR Techniques. *ACM TOIS*, 20(4), 422–446.
* Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *EMNLP*.
* Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in Data Mining: Formulation, Detection, and Avoidance. *ACM TKDD*, 6(4).
* Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS*.
* Landis, J. R., & Koch, G. G. (1977). The Measurement of Observer Agreement for Categorical Data. *Biometrics*, 33(1), 159–174.
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
