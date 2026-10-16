# Başla buradan

Bu klasör, projeyi devralan ajan için yazıldı. Önce bu sayfayı, sonra `01_TARIF.md`'yi oku. Belgeler harita
niteliğindedir, emir değildir. Mimari ve yöntem kararları senindir; yalnızca aşağıda **"kullanıcıya sor"** diye
işaretlenen konular kullanıcının kararıdır. Bir belgeyle kodun çeliştiğini görürsen kod haklıdır: belgeyi düzelt.

## 1. Görev

CV ↔ iş ilanı eşleştirme. Bir CV verildiğinde 3.000 ilanlık havuzdan en uygun ilanlar sıralanır. Sistem iki
aşamalıdır:
- **Aşama 1:** BGE-M3 bi-encoder ile BM25 sonuçları RRF ile birleştirilir ve 50 aday çıkar.
- **Aşama 2:** Bu 50 aday önce bge-reranker-v2-m3 cross-encoder ile, sonra LambdaMART ile yeniden sıralanır.

Değerlendirme sızıntısız test kümesinde ve kademeli (0–3) qrels ile yapılır.

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
| **Bitti** | Sızıntısız veri paketi (`data_bundle.zip`); notebook (`CV_Ilan_SOTA_Egitim_ve_Test_A100.ipynb`, `scripts/07_make_notebook.py` üretir); ortak çekirdek (`match_candidate_sota.py`); CPU kuru çalıştırma (OK); savunma rehberi; Faz 1/2 raporları |
| **Hiç yapılmadı** | GPU'da tek bir çalıştırma. Nöral sistemlerin (Aşama 1/2) **hiçbir sonucu yok**; süre ve bellek ölçülmedi |
| **GPU'suz ölçülen** | BM25: NDCG@10 53,66 / Top-1 64,68 (test). BM25 + LambdaMART (yalnız leksikal öznitelikler): 66,05 / 72,85 (test). Kaldıraç denemeleri yalnız val'de: `reports/lexical_lever_probe_val.json` |

## 3. Bilmen gereken dört gerçek

Hepsi `reports/metric_ceilings.json` dosyasında (`scripts/12_metric_ceilings.py`). "Kâhin" ilanları gerçek dereceye
göre sıralayan, yani mükemmel bir sistem demektir.

1. **P@10 > 85 testte ulaşılamaz.** Kâhin bile P@10 = 83,38 alır. Test sorgularının %34,5'inin 10'dan az alakalı
   (grade ≥ 2) ilanı vardır. Mavi yakalı sorgularda bu oran %70,4'tür. Val'de kâhin 85,08'dir. Hedefi "tutturmak"
   için `REL_THRESHOLD`'u, kuralları ya da P@10 tanımını değiştirmek **yasaktır**; bkz. §4. Doğru yol: P@10'u
   tavanıyla birlikte raporla, P@10_capped'i (kâhin 100) yanına koy, hedefin yeniden ifadesini kullanıcıya sor.
2. **NDCG@10 > 92, ancak grade 3 ile grade 2 ayrılırsa mümkündür.** Tüm alakalı ilanları başa koyup kendi
   aralarında rastgele sıralayan bir sistem testte 91,36'da kalır (val 91,96).
   - Grade 3 şu koşulların hepsidir: aynı uzmanlık (`occ_key`), aynı yaka, kıdem farkı ≤ 1, aynı şehir ya da
     Remote ilan.
   - Aynı uzmanlık/yaka/kıdem çiftlerinin %69,9'u **yalnız lokasyon yüzünden** grade 2'de kalır.
   - Lokasyon metinden kusursuz okunur (`loc_ok` özniteliği).
   - Test sorgularının %66,4'ünde hiç grade-3 ilan yoktur.
3. **Aşama 2, aday kümesindekilerden fazlasını sıralayamaz.** BM25'in ilk 50 adayı kâhinle yeniden sıralanırsa
   NDCG@10 testte 86,95, val'de 83,15 olur. İlk 100 adayda 93,37 ve 89,71'dir. Hibrit (dense ⊕ BM25) adayların
   tavanını notebook §9.1 val üzerinde ölçer.
   - Bu tavan 92'nin altındaysa CE'yi veya LTR'yi iyileştirmek hedefe götürmez.
   - O durumda önce Aşama 1'i ya da `N_CANDIDATES`'ı ele al.
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
- hiperparametreler, `N_CANDIDATES`, RRF ağırlıkları;
- metinden türetilen yeni öznitelikler;
- çapraz uydurma (cross-fitting), damıtma, topluluk (ensemble);
- notebook ölçüm araçları ve yeniden çalıştırma stratejisi;
- hata ayıklama ve kod düzeltmeleri.

**Kullanıcıya sor**, sormadan yapma:
1. Metrik tanımını değiştirmek: `REL_THRESHOLD`, P@10'un hesabı, alakasız sorguların atlanma kuralı, hedeflerin
   yeniden ifadesi.
2. Alaka kurallarını değiştirmek (`scripts/05_build_dataset.py` → `grade_matrix()`). Tüm sayılar değişir.
3. İlan tarafının altın (`gold_*`) etiketlerini öznitelik olarak kullanmak. Şu an yasak; yalnız değerlendirme içindir.
4. CV metinlerini dış bir LLM API'sine göndermek (gizlilik), ya da LLM ile yeniden anotasyon veya üretim (maliyet).
5. Ticari kullanımı kısıtlayan lisanslı modeller (CC-BY-NC, Gemma koşulları vb.).
6. Ürün davranışını değiştiren yanlılıklar. Örnek: kNN etiket aktarımı yeni (cold) ilanlara hiç sinyal vermez.
7. Kullanıcının belirttiği GPU bütçesini aşacak deney planları.

**Asla:**
- testte seçim yapmak;
- soğuk (cold) ilanlarla eğitmek;
- val/test CV'sinden sentetik veri üretmek;
- `gold_*` veya `legacy_*` sütunlarını öznitelik yapmak;
- vitrin CV'lerini testten çıkarmak;
- sızıntı kapısını (`05` → `leakage_checks`) gevşetmek.

Ayrıntı: `03_KONTROL_LISTELERI.md` §F.

## 5. Çalışma döngüsü (Colab'ı insan çalıştırır)

1. **Ajan (sen):** değişikliği yapar. Önce kural: notebook'u elle değil `scripts/07_make_notebook.py` ile değiştir.
   Ardından `python scripts/07_make_notebook.py` ve `python scripts/08_dry_run_notebook.py --bundle data_bundle.zip
   --workdir work/dry_run --kernel <kernel>` çalıştırır ve OK almadan bırakmaz. Bu deneyin hipotezini
   `06_DENEY_GUNLUGU.md`'ye yazar.
2. **Kullanıcı:** yeni `.ipynb` ve gerekiyorsa yeni `data_bundle.zip` ile Colab A100'de **Tümünü çalıştır** der.
   Adım adım tarif: `AKADEMIK_SAVUNMA_VE_KURULUM_REHBERI.md` BÖLÜM 1.
3. **Kullanıcı geri getirir:**
   - çalıştırılmış notebook (Colab: Dosya → İndir → .ipynb);
   - Drive'daki `MyDrive/cv_ilan_sota/` klasöründen `run_summary.json`, `per_query_test.csv`, `*_training_log.csv`,
     `ablation*.csv`, `cold_start_test.csv`, `figures/`.
4. **Ajan:** bunları `runs/<YYYY-MM-DD>_<etiket>/` altına koyar (`runs/README.md`), `02` ve `03`'teki kontrolleri
   yapar, günlüğe sonucu yazar ve bir sonraki adımı seçer.

Bir tam çalıştırma tahminen 1,5–2,5 saat sürer. Bu süre GPU'da ölçülmedi. Her çalıştırma pahalıdır: önce ucuz
teşhisleri bitir (`01` adım 1–2 ve `04` §1).

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
match_candidate_sota.py       ortak çekirdek: öznitelikler, metrikler, BM25, RRF, SOTAMatcher (çıkarım), CLI, FastAPI
scripts/05_build_dataset.py   veri paketi + sızıntı kapısı; çekirdeği pakete KOPYALAR
scripts/12_metric_ceilings.py metrik tavanları → reports/metric_ceilings.json
scripts/13_lexical_lever_probe.py kaldıraç denemesi, yalnız val, GroupKFold → reports/lexical_lever_probe_val.json
data_bundle.zip               Colab'a yüklenen paket (13 MB)
runs/                         her A100 çalıştırmasının küçük çıktıları (model ağırlıkları değil)
ajan/                         bu belgeler
```

**En sık yapılan hata:** Notebook çekirdeği repodan değil **paketten** içe aktarır (`/content/data_bundle/match_candidate_sota.py`).
`match_candidate_sota.py`'yi değiştirdiysen şunları yapmadan Colab eski kodu çalıştırır:
1. `python scripts/05_build_dataset.py` ile paketi yeniden üret.
2. Yeni `data_bundle.zip`'i Colab'a yükle.
3. Aynı Colab makinesinde daha önce açılmış `/content/data_bundle` klasörünü sil, çünkü notebook mevcut klasörü
   kontrol etmeden yeniden kullanır.
