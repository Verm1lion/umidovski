# Bu proje için eğitim tarifi

Andrej Karpathy'nin *"A Recipe for Training Neural Networks"* (2019) yazısındaki yaklaşımın bu projeye uyarlanmış
hali. Ana fikir şu: sinir ağı eğitimi **sessizce** başarısız olur. Kod çökmez, sayılar makul görünür, ama yanlıştır.
Hızlı gitmenin yolu yavaş ve titiz gitmektir.

## 0. Bu projede "sessiz başarısızlık" neye benzer

Aşağıdakilerin hiçbiri hata vermez; hepsi yalnızca biraz daha kötü (ya da şüpheli derecede iyi) bir sayı üretir.

- **Önek uyumsuzluğu.** E5 ailesi `query:` ve `passage:` önekleri ister, BGE-M3 hiç önek istemez. Model değiştirip
  `prefixes()` güncellenmezse eğitim ve çıkarım farklı dağılımda çalışır.
- **Havuz ve ölçek karışıklığı.** Bütün val karar sinyalleri WARM havuzda ölçülür (2.699 ilan, WARM IDCG; cold
  ilanlar ne adayda ne qrels'te): Aşama 1/2a log'ları, `val_zero_shot_base(WARM)`, LTR, §9.1, §10.1 ve `scripts/13`.
  Test metrikleri tam havuzdadır (3.000 ilan). İkisi aynı ölçekte değildir: val 0,70 ile test 0,66 karşılaştırması
  anlamsızdır; val'i yalnız val ile kıyasla (`metric_ceilings.json` → `val|WARM|…` / `val|warm|…`). Birim de
  karışır: log'lar ve `run_summary.json` → `val_sinyalleri` 0–1, `fmt()` çıktıları ve tablolar yüzdedir.
- **Bayat paket.** Notebook çekirdeği paketten içe aktarır. Çekirdeği değiştirip paketi `05` ile yeniden üretmezsen
  Colab eski kodu çalıştırır. Çekirdek veri üreticisinin de parçasıdır: `05` veriyi onunla üretir (tam liste:
  `CLAUDE.md` kural 5), yani çekirdek değişikliği parquet'leri de değiştirebilir. `05` `data_bundle/` klasörünü, zip'i ve
  `reports/faz2_veri_ozeti.json`'u yeniden yazar; bu yüzden `05`'ten önce `data_bundle/metadata.json`'u yedekle, sonra
  `metadata.json` → `dosyalar` sha256'larını yedekle karşılaştır. Değiştiyse bu bir veri değişikliğidir: günlüğe yaz;
  bölme ya da qrels değiştiyse dur ve kullanıcıya sor (`03` §B).
  - Colab §2, `/content/data_bundle.zip`, çalışma klasörü ve `MyDrive/data_bundle.zip` arasından **en yeni** zip'i
    seçer ve açılmış `/content/data_bundle` ile karşılaştırır (`metadata.json` + çekirdek baytları); farklıysa
    klasörü silip zip'i yeniden açar (`Açılıyor: …`, aynıysa `Açılmış paket zip ile aynı: …`). Yükleme penceresi
    aynı adı ikinci kez `data_bundle (1).zip` diye kaydeder ve o okunmaz: yeni zip'i Dosyalar paneliyle tam bu adla
    koy (`00` §7).
  - Sağlama: §2'deki `paket parmak izi (sha256[:12])`, repodaki
    `sha256sum data_bundle/metadata.json match_candidate_sota.py | cut -c1-12` ile aynı olmalı. `Çekirdek modül
    sürümü` hep `1.0.0` basar; bayat paketi göstermez.
- **Hassasiyet kayması.** Accelerate'in autocast sarmalayıcısı eğitimden sonra modelde kalır. `drop_amp_wrapper` bunu
  söker; bir sürüm değişikliği bu söküşü sessizce bozabilir. fp16 dışa aktarım ağırlıkları yuvarlar. Belirti: parite
  satırında Top-10 örtüşmesinin %100'ün belirgin altında olması.
- **Yanlış negatif.** Batch içindeki başka satırın pozitifi bu anchor için alakalı olabilir. Kıdem/yaka tuzağı
  negatiflerinin 1.730/4.883'ü grade 1'dir (`data_bundle/metadata.json` → `triplet.negatif_grade_dagilimi` →
  `neg_seniority_collar_grade`; paketin sabit bir özelliğidir, notebook bunu yazdırmaz). §7'deki tanı her
  çalıştırmada yalnız grade ≥ 2 batch-içi yanlış-negatif oranını yazdırır: başka satırların pozitifleri, başka
  satırların zor negatifleri, kendi zor negatifleri ve toplam. Grade-1 tuzaklar bu sayıma girmez; "kendi zor
  negatifleri" satırı qrels filtresi yüzünden %0 olmalıdır.
- **Test'e bakarak ayar.** Her çalıştırma testi yazdırır. Test sayısına bakıp yapılandırma değiştirmek, bir sonraki
  sayının test değil "ikinci val" olması demektir. Notebook'un yazdırdığı bütün test satırları ve tabloları (§4–§12 ve §14; ör. B1/B2/B3, §8 Recall, §11 ABL/COMP, §11.1, §12), `per_query_test.csv` ve `run_summary.json` → `hedefler` / `ana_ablation` yalnız kayıttır; ana
  sistemlerin val karşılığı §10.1'dedir. Devir hazırlığında testte yapılan keşif ölçümleri repoda yoktur ve kanıt
  sayılmaz (`06` E000).
- **Kâhin tavanını unutmak.** Aşama 2'yi haftalarca ayarlayıp aday kümesinin val tavanının (§9.1
  `aday_tavani_val(WARM)`) zaten düşük olduğunu sonradan fark etmek. Ölçek için: val WARM'da BM25'in ilk 50 adayı
  kâhinle sıralansa bile NDCG@10 84,14'te kalır.

Savunma, her aşamada bir **sağlama** yapmaktır: sonucunu önceden tahmin edebildiğin, tutmazsa durduğun bir kontrol.

## 1. Veriyle bir ol (GPU gerekmez, ≈1 saat)

Model kodundan önce veriye bak. Bu projede veri kural türevidir; kuralı içselleştirmeden model ayarlamak, neyi
ödüllendirdiğini bilmeden optimize etmektir.

**Yap:**
0. Taze klonda `data_bundle/` yoktur (git'te izlenmez): repo kökünde `unzip -o data_bundle.zip` çalıştır (zip'teki
   yollar `data_bundle/` önekli olduğu için klasör kökte oluşur) ya da `05`'i çalıştır. `scripts/10`–`13` paketi
   doğrudan `data_bundle/*.parquet`'ten okur; klasör yoksa `FileNotFoundError` verir.
1. `python scripts/12_metric_ceilings.py` çalıştır, ardından `reports/metric_ceilings.json` dosyasını oku. Üç tür
   tavanı (kâhin, ikili kusursuz, BM25 aday kümesi) val/test ve (varsa) tam/WARM ayrımıyla bil. Karar ölçeği val'in
   WARM anahtarlarıdır: `aday_kumesi_tavani(BM25)` → `val|WARM|bm25_ilk{K}_kahin_yeniden_siralama` (ilk-50 / ilk-100:
   84,14 / 90,67) ve `kahin` → `val|warm|esik2` (küçük harf). İkili kusursuzun WARM sürümü yoktur (`val` / `test`).
   Özetler ve test karşılıkları `00` §3'te; test
   değerleri yalnız hedeflerin ulaşılabilirliğini raporlamak içindir.
2. `scripts/05_build_dataset.py` içindeki `grade_matrix()` fonksiyonunu oku. Bu işlev 0–3 dereceyi deterministik
   olarak üretir.
   - **Grade ≥ 2** pratikte "aynı ISCO-4 birim grubu + yaka + |kıdem| ≤ 2" demektir. Aynı ISCO-4 çiftini grade ≥ 2'den
     yalnız yaka farkı ya da |kıdem| > 2 düşürür; bunun ne kadar nadir olduğunu train'de kendin say (`gold_isco4`
     eşleşmesi × `qrels_ground_truth.parquet`).
   - **Grade 3** ek olarak aynı uzmanlık (`gold_occ_key`), |kıdem| ≤ 1 ve aynı şehir (`gold_city`, boş değil) ya da
     Remote ilan (`is_remote`) ister.
   - Bu yüzden Top-1 ve P@10 büyük ölçüde iki taraflı bir **ISCO-4 tanıma** problemidir. NDCG@10 ise ek olarak
     **uzmanlık + lokasyon** ayrımı ister.
3. 20 rastgele train CV'si seç. Her biri için en yüksek dereceli 5 ilanı ve BM25'in ilk 5'ini yan yana oku
   (`clean_cvs.parquet`, `clean_jobs.parquet`, `qrels_ground_truth.parquet`). Kendine sor:
   - Bir insan hangi ilanı seçerdi?
   - Kural neden farklı düşünüyor?
4. Vitrin CV'lerine bak (20951 baklava ustası, 10587 uçuş dispeçeri, 10511 Java geliştirici, 21029 Java stajyeri).
   Dördü de test CV'sidir: yalnız kuralı anlamak için bak, hiçbir ayarı bunlara göre yapma (`metadata.json` →
   `vitrin_kontrol`).
   - 20951'de grade-3 ilan **imkânsızdır**: aynı uzmanlıktaki iki ilan İzmir ve Hatay'dadır, aday Ankara'dadır.
   - 10587'nin train'deki tek kapsaması sentetik bir CV'dir.
5. `python scripts/13_lexical_lever_probe.py` çalıştır. ISCO, kNN ve K=100 kaldıraçlarını **yalnız val** üzerinde,
   WARM havuzda (cold ilanlar ne sıralamada ne qrels'te), GroupKFold ile ölçer (`reports/lexical_lever_probe_val.json`).
   `scripts/11` ise BM25 + LambdaMART'ı (yalnız leksikal) **testte** ölçer; sonucu `reports/local_lexical_ltr.json`'da
   hazırdır ve yalnız kayıttır. Onu yeniden çalıştırmak testi bir kez daha okumaktır; gerek yok.
6. BM25'in Top-1 hatalarını **yalnız val** üzerinde, WARM havuzda sınıflandır; testte bu analizi yapma.
   - BM25'i yerelde kur: `bm25 = core.SparseBM25([core.tokenize(t) for t in jobs.job_struct])`, skor
     `bm25.scores([core.tokenize(cv_struct)])`, cold sütunlar −∞. Top-1 hatası:
     WARM havuzda en az bir grade ≥ 2 ilanı olan val sorgularında (438 sorgu; `metric_ceilings.json` → `kahin` →
     `val|warm|esik2` → `n_queries`; `Top1` metriği alakalısı olmayan sorguları atlar) ilk ilanın derecesinin ≤ 1
     olmasıdır.
   - `grade_matrix()`'in grade-2 koşullarına göre sırayla: yaka (`gold_collar`) farklı mı → ISCO-4 (`gold_isco4`)
     farklı ve sektör (CV `gold_industry` ↔ ilan `industry`) aynı mı → ISCO-4 de sektör de farklı mı → |kıdem|
     (`gold_seniority`) > 2 mi. Altın sütunlar burada yalnız analiz içindir; asla öznitelik yapılmaz.
   - Sayımı JSON'a yaz (örn. `runs/<tarih>_bm25-top1-hatalari-val/bm25_top1_hatalari_val.json`) ve `06`'ya kaydet.
     Hazırlıkta testte keşif amaçlı ölçülen kategori oranları (`06` E000, madde 3) repoda yoktur ve hiçbir karar
     için kullanılmaz.

**Çıkış ölçütü:** Bir CV ve bir ilan gösterildiğinde dereceyi tahmin edebiliyorsun. Hedeflerden hangisinin hangi
yapısal tavana çarptığını tek cümleyle söyleyebiliyorsun.

## 2. Uçtan uca iskelet + aptal baseline'lar

İskelet hazır: notebook. Bu adımın işi, iskeletin **doğru** olduğunu ve her parçanın beklenen sayıyı ürettiğini
kanıtlamaktır.

**Sağlamalar (ilk A100 çalıştırması):**
- **B1 (BM25) deterministiktir.** Test için tam olarak şu satırı yazdırmalı (`reports/local_bm25_test.json`):
  `Baseline 1 (BM25): {'NDCG@10': 53.66, 'MRR@10': 74.03, 'P@10': 38.37, 'P@10_capped': 48.53, 'R@10': 26.59, 'R@50': 47.46, 'Top1': 64.68, 'Top1_exact(grade3)': 9.49}`.
  Bu bir eşitlik sağlamasıdır, karar sinyali değildir. Tek bir hane farklıysa veri veya çekirdek repo ile aynı
  değildir; **dur**.
- **Rastgele skor.** §6.1 tablosundaki "Rastgele skor + eski çarpanlar (SIZINTILI)" satırı saf rastgele değildir
  (rastgele skor × altın etiketli eski çarpanlar) ve yalnız eski (sektör) metrikle ölçülür; kademeli sütunları
  tasarım gereği NaN'dir. Eski metrikte yüksek çıkmalıdır (kuru çalıştırmada da öyle,
  `reports/dry_run_executed_notebook.ipynb`): eski metriğin sızıntılı olduğunun kanıtı budur. Saf rastgele sağlama
  notebook'ta yoktur; istersen val'de, WARM havuzda, `RESULTS`'a yazmadan ölç (Colab'ı kullanıcı çalıştırır: bunu
  `scripts/07_make_notebook.py`'ye örn. §10.1'den sonra bir kod hücresi olarak ekle, `07` + `08`, sonuç bir sonraki
  çalıştırmada yazdırılır):
  ```python
  R = np.random.default_rng(CFG["SEED"]).uniform(size=(len(VAL_IDS), N_JOBS))[:, WARM_COLS]
  print(fmt(eval_scores(R, VAL_IDS, cols=WARM_COLS, store=False)))
  ```
  NDCG@10 tek haneli kalmalı; belirgin biçimde yüksekse değerlendirme altyapısında hata vardır.
- **Sıfır-atış BGE-M3** Aşama 1'in başlangıç noktasıdır. §7 bunu val'de, WARM havuzda yazdırır:
  `Zero-shot base bi-encoder (VAL, WARM havuz — Aşama 1 log'larıyla aynı ölçek): {...}` (`run_summary.json` →
  `val_zero_shot_base(WARM)`, yüzde). İnce ayarlı dense'in en iyi `eval_val_graded_ndcg@10` değeri
  (`stage1_training_log.csv`; `val_sinyalleri` → `bi_val_en_iyi`, 0–1) bunu geçmiyorsa Aşama 1'de bir hata vardır;
  ayarlama yapma, hatayı bul. §10.1 `VAL_COMP` → "İnce ayar (val)" satırı aynı karşılaştırmayı, eğitimden sonra
  yeniden kodlanmış vektörlerle ve p-değeriyle verir. Sayılar log'dakilerle birebir aynı olmayabilir, çünkü eşit
  skorlar iki yolda farklı kırılır. Kuru çalıştırmada küçük rastgele modeller yüzünden işaret bile terstir (log 0,75 −
  sıfır-atış 1,29 = −0,54 puan; `VAL_COMP` +0,57; `work/dry_run/outputs/run_summary.json`); bu sağlamada kuru
  çalıştırma sayıları kullanılmaz. Hassasiyet de farklıdır: log eğitim sırasında bf16 autocast altında, sıfır-atış
  ve §10.1 fp32 hesaplanır; küçük farklarda §10.1 `İnce ayar (val)` satırına (iki taraf da fp32) ve p'ye bak. §7'nin ilk satırı ve §11'deki "Ref: Zero-shot base bi-encoder" testtir (tam havuz);
  yalnız kayıttır.
- **Sayaçlar:**
  - `paket parmak izi (sha256[:12]): {'metadata.json': 'b0eb93042e32', 'match_candidate_sota.py': 'd6a16586ffd7'}`
    (şu anki paket; paket ya da çekirdek değişince `sha256sum` ile yeniden hesapla)
  - `MNRL satırı: 4883 … CoSENT çifti: 9719`
  - `train anchor: 2786 | val sorgu: 457 | test sorgu: 461 (… 8)`
  - `LTR çerçeveleri: {'val': (22850, 38), 'test': (23050, 38)}`

  Biri farklıysa veri veya kod değişmiştir.
- **Parite satırı.** Paketten yeniden yüklenen model, notebook'un test sıralamalarıyla örtüşmeli. Örtüşme ≈ %100,
  Top-1 aynılığı ≈ %100 olmalı. Kuru çalıştırmada küçük modellerle %98,8 / %100 çıktı. Gerçek modelle %95'in altı
  araştırılmalı (`05` §A).

Tam sağlık tablosu `02` §3'tedir.

**Aptal baseline eklemek istersen:** "Popülerlik" sıralayıcısı: her **WARM** ilanı, train CV'lerinden kaçında grade ≥ 2
aldığına göre sırala. Sayım yalnız train qrels × WARM ilanlardan yapılır (94.522 çift); cold ilanlara 0 verilir
(`scripts/13`'teki kNN'in `G[:, cold] = 0.0` satırı gibi). Cold ilanların train etiketleri sayıma girerse cold kuralı
(`03` §F) bozulur. Sorgudan bağımsız bir alt sınırdır; val'de WARM havuzda ölç (`cols=WARM_COLS`).
Soğuk-başlangıç tablosuna konmaz, çünkü orada bütün skorlar 0'dır. Bu baseline'ın üstünde kalmayan bir model gerçekte
sorguyu kullanmıyordur.

## 3. Önce aşırı öğren (kapasite var mı?)

Eğitilen model kendi eğitim verisini ezberleyemiyorsa sorun düzenlileştirme değil hatadır: veri hattı, kayıp,
örnekleyici ya da önek.

**Aşama 1 kapasite testi.** Bunu `scripts/07_make_notebook.py`'de §7.2'nin arkasına bir kod hücresi olarak ekle
(`07` + `08`); kullanıcı bir sonraki çalıştırmada koşar, sonuç yazdırılır ve günlüğe girer. Bu bölümdeki diğer
denemeler (§2 rastgele skor, Aşama 2a kapasite testi, §5'teki RRF denemesi) de aynı yolla eklenir:
```python
ids = [a for a in anchors[:300]]                       # eğitimde görülmüş train anchor'ları
Eq = encode(BI, cv_texts(ids, QP))
print("train (in-sample):", fmt(eval_scores((Eq @ E_jobs.T)[:, WARM_COLS], ids, cols=WARM_COLS, store=False)))
```
- Bu sayı val'deki en iyi `eval_val_graded_ndcg@10` değerinden belirgin biçimde yüksek olmalı (`fmt` yüzde verir,
  log 0–1).
- Train ve val **ikisi de** düşükse model öğrenemiyordur (yetersiz öğrenme). Bu durumda önce şunlara bak: LR, τ, gerçek
  adım sayısı (`stage1_training_log.csv` → `step` sütununun en büyük değeri; satır sayısı adım sayısı değildir: train
  loss 10 adımda bir yazılır (`logging_steps=10`), her değerlendirme ve sondaki eğitim özeti ayrı birer satırdır),
  kayıp eğrisi.
- Train yüksek, val düşükse genelleme sorunu vardır. 4. adıma geç.

**Aşama 2a kapasite testi.** Aynı mantıkla 200 train anchor'unun `cand_tr` listeleri CE ile yeniden sıralanır ve
NDCG ölçülür: §9'dan sonra `GradedRerankEvaluator(tr_anchor[:200], fused_tr[:200])(CE)`. Anahtar `val_` önekli olsa da
ölçüm train'dedir (0–1); `ce_val_en_iyi` ile kıyasla.

**LambdaMART.** Val üzerinde in-sample NDCG (`ltr_rank_eval(RANKER, ltr_tr, None, None)`, WARM, 0–1) ile
`ltr_cv_ndcg10` (kat dışı; `run_summary.json` → `val_sinyalleri` → `ltr_cv_ndcg10(val,WARM)`) arasındaki fark aşırı
uyumun ölçüsüdür. `ltr_cv_ndcg10`, `LTR_TRIALS` denemenin en iyisidir, yani hafifçe iyimserdir. §10.1'deki "S2: …
(kat dışı)" satırı seçilen parametrelerle yeniden kurulan kat dışı sıralamadır.

**Çıkış ölçütü:** Her öğrenen bileşen eğitim verisinde baseline'ı açıkça geçiyor.

## 4. Düzenlileştir (genelleme)

Sırayla, her seferinde tek bir değişiklik:
1. **Daha fazla gerçek sinyal.**
   - Aşama 1 yalnız 4.883 MNRL satırı (2.786 anchor) görür. Train qrels ise 94.522 WARM grade ≥ 2 çift sunar (3.689'u
     grade 3).
   - `build_mnrl_rows` içinde QRELS'ten ek pozitif satırlar eklenebilir (yalnız WARM ilanlar). Paketi yeniden
     üretmek gerekmez; değişiklik `scripts/07_make_notebook.py`'de yapılır, ardından `07` + `08` (`03` §B).
   - Grade-2 pozitifleri gürültülüdür (kör doğrulamada 43/90'ı 0–1 bulundu, `reports/qrels_validation.json`). Grade
     3'e ağırlık ver ya da grade 2'yi sınırla.
2. **Sentetik verinin katkısını ölç.** `USE_SYNTHETIC=False` ile bir ablation yap; kararı §10.1 val tablosuyla ver
   (iki çalıştırmanın `per_query_val.csv`'si, eşleştirilmiş randomizasyon). Kabul edilen 710 sentetik CV'nin 54'ünün
   uzmanlığında mevcut bölmede artık ≥ 2 gerçek train CV'si var (`05` §B). Yeni sentetik veri ya da LLM ile yeniden
   anotasyon kullanıcı kararıdır (maliyet, `00` §4 madde 3); üretim yalnız ilan tarafındaki WARM bilgiyi kullanır, hedefler teste bakılarak seçilmez.
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
  - §9.1, hibrit adayların val tavanını WARM havuzda yazdırır: `Aday kümesi tavanı (VAL, WARM havuz, …)`
    (`run_summary.json` → `tani` → `aday_tavani_val(WARM)`, yüzde). Eşik (`04` §1 düğüm 3): < 92 → Aşama 2 işine
    başlama, önce Aşama 1 / `N_CANDIDATES` / RRF; 92–95 → gri bölge, Aşama 1 / aday iyileştirmesi muhtemelen hâlâ en
    verimlisi; ≥ 95 → Aşama 2 işi karşılık verebilir. Kıyas: val WARM BM25 ilk-50 84,14 / ilk-100 90,67
    (`metric_ceilings.json` → `val|WARM|bm25_ilk{K}_kahin_yeniden_siralama`; hücre bunları `Kıyas (val, WARM, kâhin
    NDCG@10): …` satırında yazdırır). Val tavanları test
    tavanlarından birkaç puan düşüktür; test tavanı (`aday_tavani_test(tam_havuz,yalniz_rapor)`) yalnız rapor içindir.
  - Tavan düşükse `N_CANDIDATES`'ı artır (50 → 100 hesap maliyetini kabaca ikiye katlar), RRF `k` ve kanal
    ağırlıklarını val WARM R@50/R@100'e göre ayarla, ya da dense geri getirmeyi güçlendir.
  - Val R@50/R@100: `per_query_val.csv` → `sistem == "S1: Fine-tuned bi-encoder + BM25 (RRF)"` satırlarının `R@50` /
    `R@100` ortalaması. §8'deki `Recall@k` satırları testtir, yalnız kayıttır. Bir RRF ayarını §10.1 ile aynı ölçekte
    denemek için (§10.1'den sonraki bir hücre olarak): `evaluate(warm_ranked(hybrid(S_DENSE["val"], S_BM25["val"])[0], VAL_IDS),
    cols=WARM_COLS, store=False)` (`R@50`, `R@100` dahil, 0–1).
  - `core.rrf_scores` ağırlık alır, ama `SOTAMatcher.match` onu ağırlıksız çağırır ve `EXPORT_CFG`'de ağırlık yoktur.
    Ağırlığı değiştirirsen üçünü birlikte yap: `CFG`'ye bir anahtar (örn. `RRF_WEIGHTS`) ekleyip `hybrid()`'de kullan;
    `EXPORT_CFG`'ye `"rrf_weights"` olarak yaz; `SOTAMatcher.match`'te `weights=cfg.get("rrf_weights")` ile kullan
    (çekirdek değişikliği: `05` ile paketi yeniden üret, `03` §B). Yoksa paket farklı adaylar ve farklı
    `rrf` / `rrf_rank_log` öznitelikleri üretir; parite düşer ve bu düşüş yanlışlıkla fp16'ya yorulur.
- **Önemli düğmeler:** `BI_LR`, τ (`TEMP_GRID`), `BI_EPOCHS`, `N_CANDIDATES`, `CE_DOCS_PER_QUERY`, `CE_LR`, `CE_MAX_LEN`.
  Gerisi ikinci derecedendir. Izgara yerine rastgele arama kullan (Bergstra & Bengio, 2012).
- **τ taraması bilgi vermeyebilir.** Tarama yarım veriyle (2.441 satır), 1 epoch, en az 39 adım (batch 64) ve CoSENT
  olmadan yapılır. τ'yu sabitlemek ya da taramayı uzatmak meşru bir seçimdir.
- **Her ayarlama kararı yalnız şu sinyallerle verilir** (val, WARM havuz; liste `03` §A):
  - `eval_val_*` (Aşama 1 ve 2a log'ları: `stage1_training_log.csv`, `stage2a_training_log.csv`);
  - `val_zero_shot_base(WARM)` (§7, Aşama 1'in başlangıç noktası);
  - `ltr_cv_ndcg10` (§10, kat dışı GroupKFold; katlar CV grubuna (`group_id`) göre, `scripts/13` ile aynı);
  - §9.1 `aday_tavani_val(WARM)`;
  - §10.1 `VAL_TBL` / `VAL_COMP` (`val_pipeline.csv`, `val_component_tests.csv`) ve `per_query_val.csv` (hata kovaları;
    iki çalıştırma arasında `core.paired_randomization_test` ile eşleştirilmiş p);
  - `scripts/13` (`reports/lexical_lever_probe_val.json`).

  §10.1 hafifçe iyimserdir (val, erken durdurma / checkpoint ve LTR hiperparametre seçiminde kullanıldı):
  değişiklikleri birbiriyle kıyaslamak içindir, yansız kestirim değildir. Notebook sinyallerinin özeti
  `run_summary.json` içindedir: `val_sinyalleri` (`bi_val_en_iyi` / `ce_val_en_iyi` yalnız en iyi değerler, 0–1;
  `ltr_cv_ndcg10(val,WARM)`; `temperature_sweep`; `secilen_olcek`), `val_zero_shot_base(WARM)`, `tani` →
  `aday_tavani_val(WARM)`, `val_tam_hat(WARM)`, `val_bilesen_testleri`. Eğrilerin tamamı log CSV'lerindedir;
  `ltr_cv_ndcg10` adı `metrics.json`'da geçer. `scripts/13` ölçümleri notebook'a girmez. Drive'daki
  `MyDrive/cv_ilan_sota/` her çalıştırmada üzerine yazılır: iki çalıştırmayı kıyaslamak için dosyaları bir sonraki
  çalıştırmadan önce `runs/<tarih>_<etiket>/` altına al.

## 6. Son damlaları sık

Ancak 1–5 bittiyse şunlar sırayla düşünülebilir:
- topluluk (birden çok CE tohumu veya modeli, LambdaMART'a ayrı öznitelik olarak);
- daha büyük modeller (Qwen3-Reranker, bge-reranker-v2-gemma, Qwen3-Embedding);
- CE → bi-encoder damıtma (Margin-MSE);
- daha uzun bağlam (`MAX_SEQ_LEN` / `CE_MAX_LEN` 768–1024, önce kırpılma oranını §9.1'de ölç);
- çapraz uydurulmuş LTR (train sorgularında, kat dışı dense/CE öznitelikleriyle). Kanıtı yok: `06` E000'daki test
  keşfi kanıt sayılmaz; önce val'de ölç.

Maliyet ve kanıt düzeyleri `04` §3'te. Her biri bir tam çalıştırma ya da daha pahalısıdır.

## İlkeler

1. **Tek seferde tek değişiklik.** İki şeyi birden değiştirirsen hangisinin işe yaradığını bilemezsin.
2. **Önce hipotez, sonra deney.** `06_DENEY_GUNLUGU.md`'ye ne beklediğini **çalıştırmadan önce** yaz.
3. **Test mühürlü zarftır.** Her çalıştırma test yazdırır; sen yalnız kaydet. Kararı val'e göre ver (§5'teki liste).
   Raporda testin kaç kez okunduğunu (`run_summary.json` → `test_degerlendirme_sayisi`, çalıştırma sayısı) ve devir
   hazırlığındaki test temaslarını (`06` E000) dürüstçe yaz.
4. **Tavanla konuş.** Her sonucu kendi bölümünün tavanıyla ver. Val (karar): "NDCG@10 = X (aday tavanı
   `aday_tavani_val(WARM)` = Y; val WARM BM25 ilk-50 kâhin 84,14)". Test (rapor): "NDCG@10 = X (tavan 100; ikili
   kusursuz 91,36; aday tavanı `aday_tavani_test(tam_havuz,yalniz_rapor)` = Y)."
5. **Kanıt yoksa iddia yok.** Her sayı bir dosyaya (JSON/CSV) dayanmalı. "Muhtemelen" ile başlayan cümleler deney
   önerisidir, sonuç değildir.
6. **Kahraman olma.** Büyük model ve karmaşık kayıplar en sona kalır. Bu verinin yapısı (ISCO-4, uzmanlık, lokasyon)
   basit özniteliklerle çok şey kazandırıyor; bunu `scripts/13`'te (val, WARM) gör.
