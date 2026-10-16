# Başla buradan

Bu klasör, projeyi devralan ajan için yazıldı. Önce bu sayfayı, sonra `01_TARIF.md`'yi oku. Belgeler harita
niteliğindedir, emir değildir. Mimari ve yöntem kararları senindir; yalnızca aşağıda **"kullanıcıya sor"** diye
işaretlenen konular kullanıcının kararıdır. Bir belgeyle kodun çeliştiğini görürsen kod haklıdır: belgeyi düzelt.

## 0. On satırda

1. **Hedefler** (test): Top-1 > 88, NDCG@10 > 92, P@10 > 85. P@10 > 85 bu tanımla **ulaşılamaz** (kâhin 83,38):
   raporla, yeniden ifadesini kullanıcıya sor (§3).
2. **GPU'da hiç çalıştırma yok.** İlk iş: kullanıcıya notebook'u varsayılan CFG ile çalıştırtmak (`06` E001) ve
   sağlık satırlarını okumak (`02` §3).
3. **Karar yalnız val + WARM havuz:** §7 val sıfır-atış, Aşama 1/2a log'ları, §9.1 aday tavanı, §10.1 val tablosu,
   `scripts/13`. Test sayıları yalnız kayıttır.
4. **İlk teşhis §9.1 val aday tavanıdır.** < 92 ise Aşama 2'ye (CE, LTR) dokunma; önce adaylar.
5. **NDCG@10 > 92'nin yolu grade 3'ten geçer:** aynı uzmanlık + aynı şehir ya da Remote. İkili kusursuz bile 91,36.
6. **GPU'suz, val'de kanıtlı en ucuz kaldıraçlar:** ISCO-4 eşleşme özniteliği (NDCG@10 +4,8, Top-1 +5,0), kNN etiket
   aktarımı (kullanıcı kararı), 100 aday (`04` §2).
7. **Notebook'u `scripts/07` üretir, `scripts/08` doğrular;** çekirdek değişince önce `scripts/05`.
8. **Her deney:** önce hipotez (`06`), tek değişiklik, sonuç bir dosyada.
9. **Asla:** testte seçim, cold ilanla eğitim, `gold_*`/`legacy_*` öznitelik, sızıntı kapısını gevşetmek.
10. **Kullanıcıya sor:** metrik/kural/bölme değişikliği, LLM ve maliyet, lisans, ürün yanlılığı, GPU bütçesi (§4).

## 1. Görev

CV ↔ iş ilanı eşleştirme. Bir CV verildiğinde 3.000 ilanlık havuzdan en uygun ilanlar sıralanır. Sistem iki
aşamalıdır:
- **Aşama 1:** BGE-M3 bi-encoder ile BM25 sonuçları RRF ile birleştirilir ve 50 aday çıkar.
- **Aşama 2:** Bu 50 aday önce bge-reranker-v2-m3 cross-encoder ile, sonra LambdaMART ile yeniden sıralanır.

Değerlendirme sızıntısız test kümesinde ve kademeli (0–3) qrels ile yapılır. Kararlar ise yalnız val'de ve WARM
havuzda (301 cold ilan hariç) verilir; test yalnız son ölçüm ve kayıt içindir (§4).

**Kullanıcının hedefleri (test, tam havuz):** Top-1 > 88, NDCG@10 > 92, P@10 > 85. Notebook'taki `TARGETS`
sözlüğü ayrıca P@10_capped > 85'i de kontrol eder.

Senden beklenen:
1. Notebook'un Colab A100'de çalışmasını sağlamak; çalıştırmayı kullanıcı yapar, bkz. §5.
2. Sonuçları doğrulamak.
3. Hedef tutmazsa nedenini kanıtla bulup düzeltmek.
4. Ulaşılamayan bir hedefi gizlemeden, gerekçesi ve sayısıyla raporlamak.

## 2. Durum (2026-10-16)

| | |
|---|---|
| **Bitti** | Sızıntısız veri paketi (`data_bundle.zip`); notebook (`CV_Ilan_SOTA_Egitim_ve_Test_A100.ipynb`, 55 hücre, `scripts/07_make_notebook.py` üretir); ortak çekirdek (`match_candidate_sota.py`); CPU kuru çalıştırma (OK, `reports/notebook_dogrulama.json`); savunma rehberi; Faz 1/2 raporları |
| **Hiç yapılmadı** | GPU'da tek bir çalıştırma. Nöral sistemlerin (Aşama 1/2) **hiçbir sonucu yok**; süre ve bellek ölçülmedi |
| **GPU'suz ölçülen** | Test, yalnız kayıt ve sağlama için: BM25 NDCG@10 53,66 / Top-1 64,68 (`reports/local_bm25_test.json`; notebook B1 53,66'yı birebir vermeli); BM25 + LambdaMART (yalnız leksikal öznitelikler) 66,05 / 72,85 (`reports/local_lexical_ltr.json`). Kaldıraç denemeleri yalnız val'de, WARM havuzda: `reports/lexical_lever_probe_val.json` (`scripts/13`). Devir hazırlığında testte yapılan leksikal keşif ölçümleri repoda yoktur ve kanıt sayılmaz (`06` E000) |

## 3. Bilmen gereken dört gerçek

1–3. maddelerin sayıları `reports/metric_ceilings.json` dosyasındadır (`scripts/12_metric_ceilings.py`); hibrit aday
tavanını notebook §9.1 ölçer; 4. maddenin kaynağı `reports/qrels_validation.json`'dır. "Kâhin" ilanları gerçek
dereceye göre sıralayan, yani mükemmel bir sistem demektir. Test değerleri yalnız hedeflerin ulaşılabilirliğini
raporlamak içindir; kararlar val değerleriyle verilir.

1. **P@10 > 85 testte ulaşılamaz.** Kâhin bile P@10 = 83,38 alır. Test sorgularının %34,5'inin 10'dan az alakalı
   (grade ≥ 2) ilanı vardır. Mavi yakalı sorgularda bu oran %70,4'tür. Val'de kâhin 85,08'dir (WARM havuzda, yani
   §10.1 ölçeğinde 83,22). Hedefi "tutturmak" için `REL_THRESHOLD`'u, kuralları ya da P@10 tanımını değiştirmek
   **yasaktır**; bkz. §4. Doğru yol: P@10'u tavanıyla birlikte raporla, P@10_capped'i (kâhin 100) yanına koy,
   hedefin yeniden ifadesini kullanıcıya sor.
2. **NDCG@10 > 92, ancak grade 3 ile grade 2 ayrılırsa mümkündür.** Tüm alakalı ilanları başa koyup kendi
   aralarında rastgele sıralayan bir sistem testte 91,36'da kalır (val 91,96).
   - Grade 3 şu koşulların hepsidir: aynı uzmanlık (`occ_key`), aynı yaka, kıdem farkı ≤ 1, aynı şehir ya da
     Remote ilan.
   - Val'de aynı uzmanlık/yaka/kıdem çiftlerinin %71,3'ü **yalnız lokasyon yüzünden** grade 2'de kalır.
   - Lokasyon metinden kusursuz okunur: her CV'de metinden okunan şehir (`text_city`) altın şehirle aynıdır
     (`clean_cvs.parquet`). Öznitelik: `loc_ok` = Remote ya da aynı şehir (`match_candidate_sota.py`).
   - Val sorgularının %68,3'ünde hiç grade-3 ilan yoktur.
3. **Aşama 2, aday kümesindekilerden fazlasını sıralayamaz.** Karar ölçeğinde (val, WARM havuz) BM25'in ilk 50
   adayı kâhinle yeniden sıralanırsa NDCG@10 84,14 olur; ilk 100 adayda 90,67, ilk 200'de 94,58
   (`aday_kumesi_tavani(BM25)` → `val|WARM|bm25_ilk{K}_kahin_yeniden_siralama`). Val tavanları test tavanlarından
   birkaç puan düşük çıkar; val'i val ile kıyasla. Hibrit (dense ⊕ BM25) adayların tavanını notebook §9.1 val'de,
   WARM havuzda ölçer (cold ilanlar ne adayda ne qrels'te): `Aday kümesi tavanı (VAL, WARM havuz, …)` satırı,
   `run_summary.json` → `tani` → `aday_tavani_val(WARM)`. Test tavanı (`aday_tavani_test(tam_havuz,yalniz_rapor)`)
   yalnız rapor içindir. Hibrit tavanı bu WARM tabanlarla (ilk-50 84,14 / ilk-100 90,67) kıyasla; §9.1 hücresi bu
   tabanları ve eşiği `Kıyas (val, WARM, kâhin NDCG@10): …` satırında yazdırır. Eşik (`04` §1 düğüm 3):
   - < 92: Aşama 2 işine (CE, LTR) başlama; önce Aşama 1 / `N_CANDIDATES` / RRF.
   - 92–95: gri bölge; Aşama 1 / aday kümesi iyileştirmeleri muhtemelen hâlâ en verimli iştir.
   - ≥ 95: Aşama 2 işi karşılık verebilir.
4. **Etiketler kural türevidir ve üst derecelerde cömerttir.** Kör çift düzeyi doğrulamada (n = 360, LLM yargıç,
   rubrik verildi) şu bulundu:
   - Kural grade-3 çiftlerinin 47/90'ı yargıçça 2 bulundu.
   - Kural grade-2 çiftlerinin 43/90'ı 0–1 bulundu.
   - Kural grade-0 çiftlerinin hiçbiri ≥ 2 bulunmadı, ama bu çiftler rastgele seçilmişti.

   Hedefler bu yüzden "insan yargısına uyumu" değil, **kural uyumunu** ölçer. Sonuçları yorumlarken bunu söyle.
   Kaynak: `reports/qrels_validation.json`, `reports/FAZ2_VERI_RAPORU.md` §3.1.

## 4. Karar hakları

**Sen karar verirsin**, hepsi yalnız train/val kullanarak:
- model seçimi (`BI_ENCODER_BASE`, `CROSS_ENCODER_BASE`, başka aileler);
- kayıplar, örnekleyiciler, zor negatifler;
- hiperparametreler, `N_CANDIDATES`, RRF ağırlıkları (ağırlık değişikliği `EXPORT_CFG`'ye ve `SOTAMatcher.match`'e
  de girmeli; bu bir çekirdek değişikliğidir, §7. Yoksa dışa aktarılan paket notebook'tan farklı sıralar);
- metinden türetilen yeni öznitelikler;
- çapraz uydurma (cross-fitting), damıtma, topluluk (ensemble);
- notebook ölçüm araçları ve yeniden çalıştırma stratejisi;
- hata ayıklama ve kod düzeltmeleri.

**Karar sinyalleri** yalnız val'dir ve WARM havuzda ölçülür: Aşama 1/2a log'ları (`eval_val_*`),
`val_zero_shot_base(WARM)`, `ltr_cv_ndcg10`, §9.1 `aday_tavani_val(WARM)`, §10.1 `VAL_TBL` / `VAL_COMP` /
`per_query_val.csv`, `scripts/13` (liste: `03` §A). §10.1 hafifçe iyimserdir (val, erken durdurma ve LTR
hiperparametre seçiminde kullanıldı): değişiklikleri birbiriyle kıyaslamak içindir, yansız kestirim değildir. Hata
kovaları `per_query_val.csv`'den kurulur (kovalamak için CV altın sütunları değerlendirme tarafında serbesttir).
Notebook'un yazdırdığı bütün test satırları ve tabloları (§4–§12 ve §14; ör. B1/B2/B3, §8 Recall, §11 ABL/COMP, §11.1, §12), `per_query_test.csv` ve
`run_summary.json` → `hedefler` / `ana_ablation` yalnız kayıttır.

**Kullanıcıya sor**, sormadan yapma:
1. Metrik tanımını değiştirmek: `REL_THRESHOLD`, P@10'un hesabı, alakasız sorguların atlanma kuralı, hedeflerin
   yeniden ifadesi.
2. Alaka kurallarını (`scripts/05_build_dataset.py` → `grade_matrix()`) ya da bölmeyi (train/val/test üyeliği) değiştirmek.
   Tüm sayılar değişir; B1 ve önceki çalıştırmalar geçersizleşir.
3. CV metinlerini dış bir LLM API'sine göndermek (gizlilik); yeni sentetik veri ya da LLM ile yeniden anotasyon
   (maliyet). Üretim yalnız ilan tarafındaki WARM bilgiyi kullanır; hedef uzmanlıklar teste bakılarak seçilmez.
4. Ticari kullanımı kısıtlayan lisanslı modeller (CC-BY-NC, Gemma koşulları vb.).
5. Ürün davranışını değiştiren yanlılıklar. Örnek: kNN etiket aktarımı yeni (cold) ilanlara hiç sinyal vermez.
6. GPU bütçesi (A100 saati / tam çalıştırma sayısı). Henüz kayıtlı değil: ilk deney planından önce kullanıcıya sor,
   cevabı `06_DENEY_GUNLUGU.md` → "Kullanıcı kararları kaydı"na yaz. Bu bütçeyi aşan deney planı yapma.

**Asla:**
- testte seçim yapmak (test sayıları yalnız kayıttır, yukarıya bak);
- cold ilanları bir eğitim ya da seçim sinyaline sokmak;
- val/test CV'sinden sentetik veri üretmek;
- `gold_*` veya `legacy_*` sütunlarını (ilan tarafı dahil) öznitelik yapmak. Bu kullanıcıya seçenek olarak
  sunulmaz; kullanıcı ısrar ederse ayrı, açıkça etiketlenmiş bir sistem olur, asla ana sonuç olmaz;
- vitrin CV'lerini testten çıkarmak;
- sızıntı kapısını (`05` → `leakage_checks`) gevşetmek.

Ayrıntı: `03_KONTROL_LISTELERI.md` §F.

## 5. Çalışma döngüsü (Colab'ı insan çalıştırır)

1. **Ajan (sen):** önce deneyin hipotezini `06_DENEY_GUNLUGU.md`'ye yazar (tek değişiklik, mekanizma, hangi val
   sinyali hangi yönde ne kadar oynamalı; `03` §A). Sonra değişikliği yapar; notebook'u elle değil
   `scripts/07_make_notebook.py` ile değiştirir. Ardından `python scripts/07_make_notebook.py` ve
   `python scripts/08_dry_run_notebook.py --bundle data_bundle.zip --workdir work/dry_run --kernel <kernel>`
   çalıştırır. Başarı: `08` çıkış kodu 0; `reports/notebook_dogrulama.json` → `statik` = `OK` ve `dry_run.durum` =
   `OK`; `zaman` bu çalıştırmanın saati; `notebook_sha256` = `sha256sum CV_Ilan_SOTA_Egitim_ve_Test_A100.ipynb`.
   Rapor her çıkış yolunda yeniden yazılır; `HATA`, `YAPILMADI`, `ATLANDI` ya da `CALISIYOR` (süreç öldü) başarı
   değildir. Başarıyı görmeden bırakmaz.
2. **Kullanıcı:** yeni `.ipynb` ve gerekiyorsa yeni `data_bundle.zip` ile Colab A100'de **Tümünü çalıştır** der.
   Yeni zip'i §2'den önce Dosyalar paneliyle `/content/`'e ya da Drive köküne (`MyDrive/data_bundle.zip`) koyar;
   §2 en yeni zip'i seçer ve açılmış paketten farklıysa yeniden açar (bu belgenin §7'si). Aynı makinede
   `!rm -rf /content/outputs` çalıştırır; önceki çalıştırmanın dosyalarını indirdikten sonra Drive'daki
   `MyDrive/cv_ilan_sota/figures/` klasörünü siler (yoksa eski görseller pakete ve Drive'a karışır; `02` §7,
   `runs/README.md`). Adım adım tarif: `AKADEMIK_SAVUNMA_VE_KURULUM_REHBERI.md` BÖLÜM 1.
3. **Kullanıcı geri getirir.** Drive'daki `MyDrive/cv_ilan_sota/` her çalıştırmada üzerine yazılır ve temizlenmez
   (`figures/` birleştirilir, eski PNG'ler kalır); dosyalar bir sonraki çalıştırmadan **önce** indirilir/kopyalanır:
   - çalıştırılmış notebook (Colab: Dosya → İndir → .ipynb);
   - karar için: `run_summary.json`, `*_training_log.csv`, `val_pipeline.csv`, `val_component_tests.csv`,
     `per_query_val.csv`;
   - yalnız kayıt: `per_query_test.csv`, `ablation*.csv`, `cold_start_test.csv`, `figures/`.

   §2 `Drive bağlanmadı …` yazdıysa §13 Drive'a hiçbir şey yazmaz: bu çalıştırmanın dosyaları yalnız
   `/content/outputs/` ve `/content/export_model_bundle.zip` içindedir; Drive'dakiler önceki çalıştırmanındır.
4. **Ajan:** önce dosyaların bu çalıştırmaya ait olduğunu doğrular (`run_summary.json` → `paket_parmak_izi`, `cfg`,
   `toplam_sure_dk` çalıştırılmış `.ipynb`'nin §2, CFG ve §13 çıktılarıyla aynı; bkz. `runs/README.md`). Sonra
   bunları `runs/<YYYY-MM-DD>_<etiket>/` altına koyar, `02` ve `03`'teki kontrolleri yapar, günlüğe sonucu yazar ve
   bir sonraki adımı val sinyallerine göre seçer.

Bir tam çalıştırma tahminen 1,5–2,5 saat sürer (notebook başlığı). Bu süre GPU'da ölçülmedi. İlk çalıştırmadan
sonra ölçülen değerler `run_summary.json`'dadır: `toplam_sure_dk` (CFG hücresinden §13'te `run_summary.json`
yazılana kadar; kurulum, içe aktarma ve §13'ün geri kalanı dahil değil); `hucre_sureleri` (CFG hücresi ile §13
dışa aktarım hücresi arasındaki her hücrenin ilk satırı, saniyesi ve GPU tepe belleği; kurulum, içe aktarma ve CFG
hücreleri ile `run_summary.json`'ı yazan §13 dışa aktarım hücresi, parite hücresi ve §14 kayda girmez);
`gpu_tepe_bellek_GB` (hücre tepelerinin en büyüğü). `sureler_sn` yalnız eğitim koşularını tutar (τ taraması,
`Bi-encoder`, `Cross-encoder`). Her çalıştırma pahalıdır: önce ucuz (GPU'suz) teşhisleri bitir (`01` §1 ve `04` §2).

## 6. Okuma sırası

| Belge | Ne zaman |
|---|---|
| `01_TARIF.md` | Şimdi. Projenin eğitim tarifi (Karpathy yaklaşımı) |
| `02_CALISTIRMA_VE_IZLEME.md` | İlk çalıştırmadan önce. Her bölümün beklenen çıktısı ve sağlık satırları |
| `03_KONTROL_LISTELERI.md` | Her değişiklikten ve her sonuç iddiasından önce |
| `04_HEDEF_TUTMAZSA.md` | Sonuçlar hedefin altında kalınca. Teşhis ağacı ve kaldıraçlar |
| `05_BILINEN_RISKLER_VE_ACIK_SORULAR.md` | GPU'da hiç denenmemiş kod yolları, kabul edilmiş riskler |
| `06_DENEY_GUNLUGU.md` | Her deneyde: önce hipotez, sonra sonuç |

Arka plan:
- `AKADEMIK_SAVUNMA_VE_KURULUM_REHBERI.md`: kurulum, teori, sınırlılıklar.
- `reports/FAZ2_VERI_RAPORU.md`: veri, qrels, sızıntı kontrolleri.
- `reports/FAZ1_ADLI_DENETIM_RAPORU.md`: eski sistemin hataları.

## 7. Repo haritası

```
scripts/07_make_notebook.py   notebook'un TEK kaynağı (.ipynb'yi elle düzenleme; üret)
scripts/08_dry_run_notebook.py CPU kuru çalıştırma: nbformat + AST + pyflakes + uçtan uca (küçük rastgele modeller)
                              → reports/notebook_dogrulama.json
match_candidate_sota.py       ortak çekirdek: öznitelikler, metrikler, BM25, RRF, SOTAMatcher (çıkarım), CLI, FastAPI
scripts/05_build_dataset.py   veri paketi + sızıntı kapısı; çekirdeği içe aktarıp veriyi onunla üretir ve pakete KOPYALAR
scripts/12_metric_ceilings.py metrik tavanları (val/test, tam/WARM) → reports/metric_ceilings.json
scripts/13_lexical_lever_probe.py kaldıraç denemesi, yalnız val, WARM havuz, GroupKFold → reports/lexical_lever_probe_val.json
data_bundle.zip               Colab'a yüklenen paket (13 MB)
data_bundle/                  açılmış paket; git'te izlenmez. Taze klonda: unzip -o data_bundle.zip (ya da 05);
                              scripts/10–13 buradan okur
runs/                         her A100 çalıştırmasının küçük çıktıları (model ağırlıkları değil)
ajan/                         bu belgeler
```

**En sık yapılan hata:** Notebook çekirdeği repodan değil **paketten** içe aktarır (`/content/data_bundle/match_candidate_sota.py`).
`match_candidate_sota.py`'yi değiştirdiysen:
1. `05`'ten önce `data_bundle/metadata.json`'u yedekle, sonra `python scripts/05_build_dataset.py` ile paketi yeniden
   üret. `05` veriyi çekirdekle ürettiği için (tam liste: `CLAUDE.md` kural 5) parquet'ler de değişebilir: `metadata.json` →
   `dosyalar` altındaki sha256'ları karşılaştır. Değiştiyse bu bir veri değişikliğidir: günlüğe yaz; bölme ya da
   qrels değiştiyse dur ve kullanıcıya sor (§4 madde 2).
2. Kontrol: `unzip -p data_bundle.zip data_bundle/match_candidate_sota.py | cmp - match_candidate_sota.py`.
3. §2, `CV_DATA_BUNDLE` verilmemişse `/content/data_bundle.zip`, çalışma klasörü ve `MyDrive/data_bundle.zip`
   arasından **en yeni** (değiştirilme zamanı) zip'i seçer; birden çoksa hepsini tarihleriyle yazdırır. Yeni zip'i
   bu adlarla koy: Colab'ın yükleme penceresi aynı adı ikinci kez `data_bundle (1).zip` diye kaydeder, o okunmaz.
4. `/content/data_bundle` klasörünü elle silmek gerekmez: §2 seçtiği zip'teki `metadata.json` ve çekirdek baytlarını
   açılmış klasörle karşılaştırır; farklıysa klasörü silip zip'i yeniden açar (`Açılıyor: …`), aynıysa
   `Açılmış paket zip ile aynı: …` yazar. Hiç zip yoksa yükleme penceresi yalnız paket daha önce açılmamışsa çıkar;
   aksi halde `UYARI: zip bulunamadı; …` yazılır ve açılmış klasör olduğu gibi kullanılır: parmak izini mutlaka
   karşılaştır.
5. §2'deki paket hücresinin (§2'nin ilk kod hücresi) son satırı
   `Çekirdek modül sürümü: 1.0.0 | paket parmak izi (sha256[:12]): {...}` biçimindedir (`run_summary.json` →
   `paket_parmak_izi`). Sürüm sabittir, bayat paketi göstermez. Parmak izi repodaki
   `sha256sum data_bundle/metadata.json match_candidate_sota.py | cut -c1-12` ile aynı olmalı (şu anki paket:
   `b0eb93042e32` / `d6a16586ffd7`). Değilse sonuçlar eski pakete aittir.
