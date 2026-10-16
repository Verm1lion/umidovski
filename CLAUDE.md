# CV ↔ İş İlanı Eşleştirme — ajan talimatları

Bu repo CV → ilan sıralama sisteminin sızıntısız yeniden inşasıdır. Sistem iki aşamalıdır: BGE-M3 + BM25 (RRF),
ardından bge-reranker-v2-m3 + LambdaMART. Eğitim Colab A100 notebook'unda yapılır. **İşe başlamadan önce
`ajan/00_BASLA_BURADAN.md`'yi baştan sona oku**, sonra `ajan/01_TARIF.md`'yi.

## Değişmez kurallar

1. **Test mühürlü zarftır.** Hiçbir karar test sayısına göre verilmez: τ, epoch, checkpoint, öznitelik, model ya
   da hangi çalıştırmanın raporlanacağı. Kararlar yalnız **val** sinyalleriyle, **WARM havuzda** verilir: Aşama 1/2a
   log'ları (`eval_val_*`), `val_zero_shot_base(WARM)`, `ltr_cv_ndcg10`, §9.1 `aday_tavani_val(WARM)`, §10.1
   `VAL_TBL` / `VAL_COMP` / `per_query_val.csv`, `scripts/13` (`ajan/03_KONTROL_LISTELERI.md` §A). Notebook'un yazdırdığı bütün test satırları ve tabloları (§4–§12 ve §14; ör. B1/B2/B3, §8 Recall, §11 ABL/COMP,
   §11.1, §12), `per_query_test.csv` ve `run_summary.json` → `hedefler` / `ana_ablation` yalnız kayıttır.
   Devir hazırlığında testte yapılan keşif ölçümleri repoda yoktur ve kanıt sayılmaz (`ajan/06_DENEY_GUNLUGU.md` E000).
2. **Sızıntı kapısı gevşetilmez.** Kapı `scripts/05_build_dataset.py` → `leakage_checks()` içindedir. Cold ilanlar
   (301) hiçbir eğitim veya seçim sinyaline girmez; val sinyallerinin hepsi WARM havuzda ölçülür (`WARM_COLS`; §9.1,
   §10.1 ve `scripts/13` dahil). `gold_*` ve `legacy_*` sütunları (ilan tarafı dahil) **asla** öznitelik yapılmaz;
   bu kullanıcıya sorulacak bir seçenek değildir (kullanıcı ısrar ederse ayrı, açıkça etiketlenmiş bir sistem olur,
   ana sonuç olmaz). Sentetik veri yalnız train'dedir ve val/test metninden üretilmez. Yeni sentetik veri ya da LLM
   ile yeniden anotasyon kullanıcı kararıdır (maliyet); üretim yalnız ilan tarafındaki WARM bilgiyi kullanır, hedefler
   teste bakılarak seçilmez. Vitrin CV'leri testte kalır.
3. **Metrik tanımı ve alaka kuralları kullanıcının kararıdır.** `REL_THRESHOLD`, `grade_matrix()`, P@10 hesabı ve
   hedeflerin yeniden ifadesi bu kapsamdadır. P@10 > 85 testte yapısal olarak ulaşılamaz (kâhin 83,38; etiketlerin
   yapısıdır, model yok). Bunu raporla; tanımla oynayarak "tutturma".
4. **Notebook elle düzenlenmez.** Kaynak `scripts/07_make_notebook.py`'dir. Her değişiklikten sonra:
   ```bash
   python scripts/07_make_notebook.py
   python scripts/08_dry_run_notebook.py --bundle data_bundle.zip --workdir work/dry_run --kernel <kernel>
   ```
   İş ancak şu durumda bitmiştir: `08` çıkış kodu 0; `reports/notebook_dogrulama.json` içinde `"statik": "OK"` ve
   `dry_run.durum = "OK"` (`RUN_OK: true`, `export_zip: true`); `zaman` bu çalıştırmanın saati ve `notebook_sha256`
   = `sha256sum CV_Ilan_SOTA_Egitim_ve_Test_A100.ipynb`. Rapor her çıkış yolunda yeniden yazılır: `HATA`,
   `YAPILMADI` (statik hata), `ATLANDI (--static-only)` ya da `CALISIYOR`'da kalmış (süreç öldü) bir rapor başarı değildir.
5. **Çekirdek paketin içindedir ve veri üreticisinin parçasıdır.** `05`, `match_candidate_sota.py`'yi yalnız pakete
   kopyalamaz, veriyi de onunla üretir (`tr_ascii`, `tr_lower`, `norm_city`, `parse_cv_sections` /
   `parse_seniority_text` / `parse_years_experience`, `build_cv_struct` / `build_job_struct`, `tokenize` /
   `SparseBM25` / `TitleSimilarity`); çekirdek değişikliği parquet'leri değiştirebilir. Çekirdek değişince:
   - `05`'ten önce `data_bundle/metadata.json`'u yedekle, sonra `python scripts/05_build_dataset.py` (paket + zip +
     kapı) ve `metadata.json` → `dosyalar` altındaki parquet sha256'larını karşılaştır. Değiştiyse bu bir veri
     değişikliğidir: günlüğe yaz; bölme ya da qrels değiştiyse dur ve kullanıcıya sor;
   - kontrol: `unzip -p data_bundle.zip data_bundle/match_candidate_sota.py | cmp - match_candidate_sota.py`;
   - RRF ağırlıkları gibi çıkarımı etkileyen bir ayar `EXPORT_CFG`'ye ve `SOTAMatcher.match`'e de girer (çekirdek
     değişikliği); yoksa dışa aktarılan paket notebook'tan farklı sıralar.

   Yeni zip §2'den önce Dosyalar paneliyle `/content/data_bundle.zip` olarak (ya da Drive köküne
   `MyDrive/data_bundle.zip`) konur. §2, `CV_DATA_BUNDLE` yoksa bu adaylar arasından **en yeni** zip'i seçer, açılmış
   `/content/data_bundle` ile karşılaştırır (`metadata.json` + çekirdek baytları) ve farklıysa yeniden açar
   (`Açılıyor: …`; aynıysa `Açılmış paket zip ile aynı: …`); klasör elle silinmez. Yükleme penceresi yalnız hiç zip
   yokken ve paket daha önce açılmamışken çıkar (aksi hâlde `UYARI: zip bulunamadı; …`); pencere aynı adı ikinci kez
   `data_bundle (1).zip` diye kaydeder, o okunmaz. §2'deki `paket parmak izi`
   repodaki `sha256sum data_bundle/metadata.json match_candidate_sota.py | cut -c1-12` ile aynı olmalı;
   `Çekirdek modül sürümü` (sabit `1.0.0`) bayat paketi göstermez.
6. **Her deney günlüğe yazılır.** Hipotez çalıştırmadan **önce**, sonuç sonra yazılır (`ajan/06_DENEY_GUNLUGU.md`).
   Çalıştırma çıktıları `runs/<tarih>_<etiket>/` altına konur (`runs/README.md`). Drive'daki `MyDrive/cv_ilan_sota/`
   her çalıştırmada üzerine yazılır (`figures/` birleştirilir, eski PNG'ler kalır): dosyalar bir sonraki çalıştırmadan
   **önce** kopyalanır/indirilir.
7. **Kanıtsız iddia yok.** Her sayı bir dosyaya (JSON/CSV) dayanır. NDCG için %95 GA, karşılaştırmalar için
   eşleştirilmiş randomizasyon p-değeri verilir (val'de: §10.1 `val_component_tests.csv`; iki çalıştırma arasında
   `per_query_val.csv`). Sonuçlar tavanlarıyla birlikte yazılır (`reports/metric_ceilings.json`); val kararı val
   tavanıyla kıyaslanır.

## Ortam

- Python ≥ 3.10. Notebook'un sabit sürümleri: sentence-transformers 6.1.0, transformers 5.17.0, accelerate 1.15.0,
  datasets 5.1.0, lightgbm 4.7.0, optuna 5.0.0, rank_bm25 0.2.2, faiss-cpu 1.15.1. Ayrıca torch, scikit-learn,
  pandas, pyarrow, matplotlib, seaborn, nbformat, nbclient, ipykernel, pyflakes gerekir (Colab dışında notebook paket
  kurmaz; `CV_FORCE_INSTALL=1` yalnız sabit listeyi kurar, matplotlib/seaborn onda yoktur).
- GPU yoksa yerel olarak yalnız şunlar çalışır: veri paketi (`05`), kuru çalıştırma (küçük rastgele modeller,
  `CV_DRY_RUN=1`), rapor (`scripts/10`) ve leksikal betikler (`scripts/11`, `12`, `13`).
- Taze klonda `data_bundle/` yoktur (git'te izlenmez): `scripts/10`–`13`'ten önce repo kökünde
  `unzip -o data_bundle.zip` çalıştır (klasörü oluşturur) ya da `05`'i çalıştır.
- Colab'ı kullanıcı çalıştırır. Döngü `ajan/00_BASLA_BURADAN.md` §5'te anlatılıyor.

## Hızlı başvuru

| Ne | Nerede |
|---|---|
| Tavanlar (kâhin: val/test × tam/warm/cold; BM25 aday kümesi: val/test × tam/WARM; ikili kusursuz: val/test, yalnız tam havuz) | `reports/metric_ceilings.json` (`scripts/12_metric_ceilings.py`) |
| Hibrit aday tavanı (karar: val, WARM) | notebook §9.1 → `run_summary.json` → `tani` → `aday_tavani_val(WARM)` → `NDCG@10` |
| §9.1 eşiği | `aday_tavani_val(WARM)` → `NDCG@10` < 92: Aşama 2 işine başlama, önce Aşama 1 / `N_CANDIDATES` / RRF; 92–95: gri bölge (Aşama 1 / aday iyileştirmesi muhtemelen hâlâ en verimlisi); ≥ 95: Aşama 2 işi karşılık verebilir. Kıyas (kâhin NDCG@10): val WARM BM25 ilk-50 84,14 / ilk-100 90,67 (`metric_ceilings.json` → `aday_kumesi_tavani(BM25)` → `val\|WARM\|bm25_ilk{K}_kahin_yeniden_siralama` → `NDCG@10`). Val aday tavanları test tavanlarından birkaç puan düşüktür (`ajan/04_HEDEF_TUTMAZSA.md` §1 düğüm 3) |
| Val tam hat ve bileşen testleri (karar) | notebook §10.1 → `outputs/val_pipeline.csv`, `val_component_tests.csv`, `per_query_val.csv` (`run_summary.json` → `val_tam_hat(WARM)`, `val_bilesen_testleri`). Hafifçe iyimserdir (val erken durdurma ve LTR seçiminde kullanıldı): değişiklikleri birbiriyle kıyaslamak içindir, yansız kestirim değildir |
| Kaldıraçların val ölçümü (leksikal, GPU'suz, WARM havuz) | `reports/lexical_lever_probe_val.json` (`scripts/13_lexical_lever_probe.py`) |
| BM25 / leksikal LTR test referansı (yalnız kayıt) | `reports/local_bm25_test.json`, `reports/local_lexical_ltr.json` |
| Veri ve qrels raporu | `reports/FAZ2_VERI_RAPORU.md` (`scripts/10` üretir) |
| Kurulum, teori, sınırlılıklar | `AKADEMIK_SAVUNMA_VE_KURULUM_REHBERI.md` |
| Sağlık satırları ve beklenen değerler | `ajan/02_CALISTIRMA_VE_IZLEME.md` §3 |
| Hedef tutmazsa | `ajan/04_HEDEF_TUTMAZSA.md` |

Kullanıcıya görünen metinler (belgeler, rapor, notebook) Türkçedir; ondalık ayırıcı virgüldür.
