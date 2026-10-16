# Çalıştırma ve izleme (runbook)

Kullanıcı için adım adım Colab tarifi: `AKADEMIK_SAVUNMA_VE_KURULUM_REHBERI.md` BÖLÜM 1. Bu belge **ajan** içindir:
- her bölümün ne yaptığı;
- hangi satırın sağlık göstergesi olduğu;
- sorun çıkınca hangi düğmenin çevrileceği;
- çalıştırmadan sonra neyin geri getirileceği.

Bütün süre ve bellek değerleri **tahmindir**. Notebook GPU'da hiç çalışmadı. İlk gerçek çalıştırmanın
`run_summary.json` dosyası bu tahminlerin yerini alır:
- `hucre_sureleri`: CFG hücresinden sonraki her hücrenin ilk satırı, `sn` ve `gpu_tepe_GB`. Tepe bellek her hücrenin
  başında sıfırlanır, yani değer o hücreye aittir. Kayda girmeyenler: §0 kurulum ve içe aktarma hücreleri, CFG
  hücresinin kendisi, §13 dışa aktarım hücresi (özet onun içinde yazılır), §13+ ve §14.
- `toplam_sure_dk`: CFG hücresinden `run_summary.json` yazılana kadar. `gpu_tepe_bellek_GB`: hücre tepelerinin en büyüğü.
- `sureler_sn`: yalnız eğitim koşuları (`τ=…`, `Bi-encoder`, `Cross-encoder`; `trainer.train()` süresi).

## 1. Çalıştırmadan önce (ajan)

Bu değişikliğin hipotezi `06_DENEY_GUNLUGU.md`'ye **önceden** yazılmış olmalı (`03` §A).

```bash
python scripts/07_make_notebook.py                     # notebook'u üret (elle düzenleme yok)
python scripts/08_dry_run_notebook.py --bundle data_bundle.zip --workdir work/dry_run --kernel <kernel>; echo "çıkış: $?"
cat reports/notebook_dogrulama.json                     # "statik": "OK", dry_run.durum "OK", RUN_OK true, export_zip true
sha256sum CV_Ilan_SOTA_Egitim_ve_Test_A100.ipynb        # raporun notebook_sha256 değeriyle aynı olmalı
```

Başarı yalnız şudur: `08` çıkış kodu 0; raporda `"statik": "OK"` ve `dry_run.durum = "OK"` (`RUN_OK: true`,
`export_zip: true`); `zaman` bu çalıştırmanın saati; `notebook_sha256` = yukarıdaki `sha256sum`. Rapor her çıkış
yolunda yeniden yazılır. `HATA`, `YAPILMADI` (statik hata), `ATLANDI (--static-only)` ya da `CALISIYOR`'da kalmış
(süreç öldü ya da zaman aşımı) bir rapor başarı değildir. Son kuru çalıştırma ≈110 sn sürdü (`dry_run.sure_sn`).

Çekirdek veya veri değiştiyse önce `05`'i çalıştır. Çekirdek veri üreticisinin de parçasıdır (değişmez kural 5):
1. Taze klonda `data_bundle/` yoktur (git'te izlenmez): önce `unzip -o data_bundle.zip`. Sonra
   `data_bundle/metadata.json`'u scratch'e yedekle; `05` klasörü silip yeniden yazar.
2. `python scripts/05_build_dataset.py`. Paketi, zip'i ve sızıntı kapısını yeniden üretir; kapı geçmezse paket yazılmaz.
3. `metadata.json` → `dosyalar` altındaki parquet sha256'larını yedekle karşılaştır. `05` veriyi çekirdekle üretir
   (`tr_ascii`, `tr_lower`, `norm_city`, `build_cv_struct` / `build_job_struct`, `tokenize` / `SparseBM25` /
   `TitleSimilarity`); çekirdek değişikliği parquet'leri değiştirebilir. Değiştiyse bu bir veri değişikliğidir:
   günlüğe yaz. Bölme ya da qrels değiştiyse dur ve kullanıcıya sor.
4. `unzip -p data_bundle.zip data_bundle/match_candidate_sota.py | cmp - match_candidate_sota.py` sessiz kalmalı.
5. §2'nin basacağı parmak izini hesapla: `sha256sum data_bundle/metadata.json match_candidate_sota.py | cut -c1-12`
   (şu anki paket: `b0eb93042e32` / `d6a16586ffd7`).

Kullanıcıya şunları söyle:
- hangi dosyaları yükleyeceğini (`.ipynb`, değiştiyse `data_bundle.zip`);
- `MyDrive/cv_ilan_sota/` içindeki önceki çalıştırmanın dosyalarını yeni çalıştırmadan **önce** indirmesini ya da
  kopyalamasını. Her çalıştırma bu klasörün üzerine yazar (§7);
- yeni zip'i §2'den önce Dosyalar paneliyle `/content/data_bundle.zip` olarak (ya da Drive köküne
  `MyDrive/data_bundle.zip`) koymasını. §2 en yeni zip'i seçer, açılmış paketle karşılaştırır, farklıysa yeniden açar
  (`00` §7); yükleme penceresi yalnız hiç zip yokken ve paket daha önce açılmamışken çıkar;
- aynı Colab makinesinde `!rm -rf /content/outputs` çalıştırmasını ve önceki dosyaları indirdikten sonra
  `MyDrive/cv_ilan_sota/figures/` klasörünü silmesini (yoksa önceki çalıştırmanın görselleri pakete ve Drive'a
  karışır; bu belgenin §7'si);
- §2'nin ilk hücresinin son satırındaki (`Çekirdek modül sürümü: 1.0.0 | paket parmak izi (sha256[:12]): …`) iki
  değerin yukarıdakilerle aynı olması gerektiğini.

## 2. Bölüm haritası

| § | Ne yapar | Ana CFG | Yazdığı dosyalar | Tahmini süre (A100-40GB) |
|---|---|---|---|---|
| 0 | `pip install` (sabit sürümler), GPU bayrakları, sürüm uyarısı | — | — | 1–3 dk |
| 1 | `CFG`; hücre süre/bellek kaydı (`CELL_LOG`) burada başlar | hepsi | — | — |
| 2 | Drive bağla; zip'i bul (`CV_DATA_BUNDLE` → `/content/data_bundle.zip` → çalışma klasörü → `MyDrive/data_bundle.zip`; hiçbiri ve açılmış paket yoksa yükleme penceresi); açılmış paketle karşılaştır, farklıysa silip yeniden aç; parmak izi; sızıntı assert'leri | `USE_DRIVE` | — | <1 dk |
| 3 | Metrik altyapısı (`evaluate`, `fmt`, `EVAL_LOG`) | `REL_THRESHOLD` | — | — |
| 4 | B1 BM25 (+ rank_bm25 paritesi) | — | — | <1 dk |
| 5 | B2 sıfır-atış e5-small (ilk HF indirmesi) | `BASELINE_SMALL` | — | 1–2 dk |
| 6 | B3 eski yaklaşım (e5-small + CosineSimilarityLoss); §6.1 sızıntı gösterimi | `LEGACY_*` | `figures/b3_training.png` | 4–10 dk |
| 7 | Aşama 1 hazırlık: BGE-M3 sıfır-atış (test + val WARM), dense negatif madenciliği, MNRL/CoSENT satırları, yanlış-negatif tanısı | `DENSE_MINED_NEG`, `COSENT_PER_GRADE` | — | 1–2 dk |
| 7.1 | τ taraması (yarım veri, 1 epoch, CoSENT yok) | `TUNE_TEMPERATURE`, `TEMP_GRID`, `TEMP_FRACTION` | `figures/temperature_sweep.png` | 5–10 dk |
| 7.2 | Aşama 1 tam eğitim + erken durdurma | `BI_*`, `EVALS_PER_EPOCH`, `EARLY_STOP_PATIENCE` | `stage1_training_log.csv`, `bi_encoder_best/` | 18–35 dk |
| 8 | Hibrit RRF (dense ⊕ BM25) | `RRF_K` | — | <1 dk |
| 9 | Aşama 2a cross-encoder (LambdaLoss) | `CE_*`, `N_CANDIDATES` | `stage2a_training_log.csv`, `cross_encoder_best/` | 22–35 dk |
| 9.1 | **Tanı:** aday kümesi tavanı (val, WARM; test tavanı yalnız rapor), belirteç kırpılma oranları, şimdiye kadarki en uzun 5 hücre | — | (`run_summary.json` → `tani`) | <1 dk |
| 10 | Öznitelik sınıflandırıcıları, LTR çerçeveleri, LambdaMART (Optuna 40 deneme × 5 kat) | `LTR_*` | `figures/ltr_importance.png` | 5–20 dk (CPU; Optuna ölçülmedi) |
| 10.1 | **Val tam hat** (WARM havuz, LambdaMART kat dışı) ve val bileşen testleri: karar sinyali | — | `val_pipeline.csv`, `val_component_tests.csv`, `per_query_val.csv` | <1 dk (CPU) |
| 11 | Ablation (5 ana + 3 ek sistem), bileşen testleri, HEDEF karşılaştırması (test, yalnız kayıt) | — | `ablation_test.csv`, `ablation_component_tests.csv` | <1 dk |
| 11.1 | Sektör tablosu, soğuk-başlangıç (301 cold ilan) (test, yalnız kayıt) | — | `cold_start_test.csv` | 2–3 dk |
| 12 | Vitrin hata analizi, dikkat + SHAP, Top-1 hataları (test, yalnız kayıt) | — | `figures/attention_*`, `shap_*` | 1–2 dk |
| 13 | Dışa aktarım, `run_summary.json`, `per_query_test.csv`, val CSV'lerinin pakete kopyası, zip, Drive kopyası, `flush_and_unmount` | `EXPORT_HALF` | `export_model_bundle.zip` (≈2,3 GB) | 2–6 dk |
| 13+ | Paket ↔ notebook paritesi, FastAPI duman testi | — | — | 1–2 dk |
| 14 | Özet, `RUN_OK.json` | — | `RUN_OK.json` | — |

Tablodaki aralıkların toplamı ≈ 63–136 dakikadır ("<1 dk" satırları 0–1 sayıldı); notebook başlığı "≈ 1,5–2,5 saat"
der.

§10'daki Optuna süresi ölçülmedi; kayıtlı bir ölçüm yok. `LTR_TRIALS` × `LTR_FOLDS` = 40 × 5 = 200 LightGBM eğitimi
CPU'da koşar. Her biri erken durdurmalıdır ve en çok `LTR_MAX_ROUNDS` = 2000 tur sürer. Rehberdeki "3–5 dk" de bir
tahmindir. Gerçek süreyi ilk çalıştırmanın `hucre_sureleri` kaydı verir: §10'un
`from sklearn.model_selection import GroupKFold` satırıyla başlayan hücresi.

## 3. Sağlık satırları (sırayla oku)

Sağlık satırları çalışmanın doğru koştuğunu gösterir. Kararlar yalnız **val** sinyalleriyle, **WARM havuzda**
verilir: Aşama 1/2a log'ları, `val_zero_shot_base(WARM)`, `ltr_cv_ndcg10`, §9.1 `aday_tavani_val(WARM)`, §10.1
`VAL_TBL` / `VAL_COMP` / `per_query_val.csv`, `scripts/13` (WARM, `reports/lexical_lever_probe_val.json`; GPU'suz,
notebook dışında) (`03` §A). Testten okunan satırlar ya yalnız sağlamadır (§4 B1, §11 /
§11.1 sorgu sayıları, §13+ parite) ya da yalnız kayıttır (§7'nin ilk satırı, §8 Recall, §11 / §11.1 / §12
metrikleri). Ölçek: log'lar, `ltr_cv_ndcg10` ve `per_query_val.csv` 0–1'dir; yazdırılan sözlükler ve tablolar
yüzdedir.

| Nerede | Beklenen | Değilse |
|---|---|---|
| §0 | `GPU: NVIDIA A100… \| bf16=True \| tf32=True \| fp16=False` | A100 değilse batch'leri düşür (bu belgenin §4'ü) |
| §0 | `UYARI: Sabitlenmiş sürümlerden farklı…` **yok** | Oturumu yeniden başlat, tekrar çalıştır |
| §2 | `Açılıyor: …` ya da `Açılmış paket zip ile aynı: …` | `UYARI: zip bulunamadı; …` ise önceden açılmış klasör olduğu gibi kullanıldı: parmak izine mutlaka bak |
| §2 | `Çekirdek modül sürümü: 1.0.0 \| paket parmak izi (sha256[:12]): {'metadata.json': 'b0eb93042e32', 'match_candidate_sota.py': 'd6a16586ffd7'}` (şu anki paket; §1 adım 5) ve `Sızıntı koruyucuları: TAMAM` | Parmak izi farklıysa bayat paket: dur. Sürüm sabit `1.0.0`'dır, bayat paketi göstermez |
| §2 | `İlan havuzu: {'warm': 2699, 'cold': 301} \| qrels satırı: 608691` | Veri farklı; dur |
| §2 | `train anchor: 2786 \| val sorgu: 457 \| test sorgu: 461 (… 8)` | `USE_SYNTHETIC=False` ise anchor 2076 |
| §2 | `Vitrin (test) adayları:` dört aday (20951, 10587, 10511, 21029) | Eksik varsa o CV testte değil (notebook uyarmadan atar): veri değişmiş, dur |
| §4 | `BM25 paritesi (rank_bm25 ile): TAMAM` ve B1 = **53,66 / 74,03 / 38,37 / 48,53 / 26,59 / 47,46 / 64,68 / 9,49** (test; yalnız sağlama: BM25 deterministiktir, `reports/local_bm25_test.json`) | Veri/çekirdek uyuşmazlığı; dur |
| §7 | `Zero-shot base bi-encoder (VAL, WARM havuz — Aşama 1 log'larıyla aynı ölçek): {...}` (`run_summary.json` → `val_zero_shot_base(WARM)`) | Aşama 1'in val başlangıç noktası; NDCG@10'u not al. Üstteki `Zero-shot base bi-encoder:` satırı testtir (yalnız kayıt) |
| §7 | `MNRL satırı: 4883 … CoSENT çifti: 9719` | Satır üreticisi değişmiş |
| §7 | Yanlış-negatif tanısı (grade ≥ 2; üç satır + toplam). "Kendi zor negatifleri" `%0.00` olmalı: paketteki negatiflerin hepsi grade ≤ 1 (`metadata.json` → `triplet.negatif_grade_dagilimi`), dense negatifler grade 0 | Toplam yüksekse (≳%3) yanlış-negatif maskesi düşün (`04`) |
| §7.1 | `ölçek=20.0 (τ=0.050) -> val NDCG@10=…`, `Seçilen ölçek: …` (val, WARM, 0–1) | İki değer çok yakınsa seçim gürültüdür |
| §7.2 | `stage1_training_log.csv` → `eval_val_graded_ndcg@10` (0–1) yükseliyor; en iyi değer (`val_sinyalleri` → `bi_val_en_iyi`) × 100, §7'deki val WARM sıfır-atış NDCG@10'undan yüksek (aynı metrik ve ölçek; log bf16 autocast altında, sıfır-atış fp32 hesaplanır: küçük farklarda §10.1 "İnce ayar (val)" satırına bak, orada iki taraf da fp32) | İlk değerlendirme en iyiyse LR/epoch. Sıfır-atışı geçmiyorsa ya da hiç yükselmiyorsa ayarlama yapma, hata ara (`01` §3) |
| §8 | `Recall@10/50/100 (aday tavanı)`: test, yalnız kayıt | Val karşılığı: §9.1 tavanındaki `R@50` (= §10.1 "S1" `R@50`); R@100 için `per_query_val.csv` → S1 satırlarının ortalaması. Val WARM kâhin R@50 = 94,59 (`metric_ceilings.json` → `kahin` → `val\|warm\|esik2`; çok alakalılı sorgular yüzünden 100 olamaz) |
| §9 | `CE eğitim listesi: 2786 \| ortalama liste uzunluğu: 16.0` | — |
| §9.1 | `Aday kümesi tavanı (VAL, WARM havuz, ilk 50 hibrit aday, kâhin sıralama): {...}` → NDCG@10 (`tani` → `aday_tavani_val(WARM)`) | Eşik aşağıda ("§9.1 eşiği") |
| §9.1 | Belirteç kırpılma oranları (BI ilan/CV, CE çift) | Yüksekse `MAX_SEQ_LEN` / `CE_MAX_LEN` ya da sıkıştırma (`04`) |
| §9.1 | `Şimdiye kadarki en uzun hücreler (sn, GPU tepe GB): [...]` | Süreleri §2 tahminleriyle kıyasla; bir hücrenin tepesi 40 GB'a yakınsa sonraki çalıştırmada bu belgenin §4'ü |
| §10 | `Val sektör sınıflandırıcı doğruluğu: %… \| yaka: %…` (kayıtlı beklenen değer yok; ilk çalıştırmanın değerini günlüğe yaz, sonraki çalıştırmaları onunla kıyasla) | Önceki çalıştırmadan belirgin düşükse sklearn sürümü / veri farkı |
| §10 | `LTR çerçeveleri: {'val': (22850, 38), 'test': (23050, 38)} \| özellik sayısı: 35` | `N_CANDIDATES` değiştiyse satırlar ona göre |
| §10 | `En iyi CV NDCG@10 (val, GroupKFold…)` (0–1; `val_sinyalleri` → `ltr_cv_ndcg10(val,WARM)`) | §10.1 "S2 … (kat dışı)" NDCG@10 / 100 buna yakın olmalı: aynı katlar, aynı parametreler; CV değeri 40 denemenin en iyisi olduğu için biraz iyimserdir. Belirgin fark: Optuna CV gürültüsüne uymuş olabilir. Test S2 ile kıyaslama yapılmaz |
| §10.1 | `VAL (WARM havuz) — karar sinyali; …` ve altı satırın hepsinde `sorgu = 438` | Farklıysa val değerlendirme kümesi değişmiş (438 = alakalı WARM ilanı olan val sorgusu; `metric_ceilings.json` → `kahin` → `val\|warm\|esik2` → `n_queries`) |
| §10.1 | `VAL_COMP`: "İnce ayar (val)", "BM25 füzyonu (val)", "Cross-encoder (val)", "LambdaMART (val, kat dışı)" → ΔNDCG@10, ΔTop1, p | Δ ≤ 0 ya da p büyükse o bileşen val'de katkı vermiyor: `04` §1. "İnce ayar (val)" ≤ 0 ise Aşama 1'de hata ara (`01` §3). Sayılar hafifçe iyimserdir (val erken durdurmada, checkpoint ve LTR hiperparametre seçiminde kullanıldı): değişiklikleri birbiriyle kıyasla, yansız kestirim sayma |
| §11 | Ablation tablosunda `sorgu = 453` her satırda | Farklıysa değerlendirme kümesi değişmiş |
| §11 | `Hedef karşılaştırması` satırları (`run_summary.json` → `hedefler`) | Yalnız kayıt ve rapor. Sonraki adım val sinyalleriyle seçilir (`04`). P@10 için test kâhin tavanı 83,38 (etiketlerin yapısı) |
| §11.1 | Soğuk-başlangıç tablosu `sorgu = 360` | — |
| §13 | `run_summary.json yazıldı \| test değerlendirme sayısı: 14 \| toplam süre: … dk` (kuru çalıştırmada da 14, `reports/dry_run_executed_notebook.ipynb`) | 14 değilse testte yeni bir ölçüm eklenmiş: nedenini günlüğe yaz |
| §13 | `Paket: … (≈2.3 GB)`, `Drive yedeği senkronize edildi` | Drive kotası dolu olabilir; kopyalama try içinde değil |
| §13+ | `Paket ↔ notebook paritesi (30 test sorgusu): … örtüşme ≈ %100, Top-1 aynı ≈ %100` | <%95 ise hassasiyet kayması (`05` §A) |
| §13+ | `FastAPI POST /match -> 200` | fastapi kurulu değilse atlanır (sorun değil) |

**§9.1 eşiği** (`04` §1 düğüm 3; WARM val tavanının NDCG@10'u):
- **< 92:** Aşama 2 işine başlama. Önce Aşama 1 / `N_CANDIDATES` / RRF.
- **92–95:** gri bölge. Aşama 1 / aday iyileştirmesi muhtemelen hâlâ en verimlisidir.
- **≥ 95:** Aşama 2 işi karşılık verebilir.

Kıyas: val WARM BM25 ilk-50 84,14 / ilk-100 90,67 (`reports/metric_ceilings.json` → `aday_kumesi_tavani(BM25)` →
`val|WARM|bm25_ilk{K}_kahin_yeniden_siralama`); hücre bu tabanları ve eşiği `Kıyas (val, WARM, kâhin NDCG@10): …`
satırında yazdırır. Val tavanları test tavanlarından birkaç puan düşüktür.
`aday_tavani_test(tam_havuz,yalniz_rapor)` yalnız rapor içindir.

## 4. Bellek yetmezse (OOM)

Tahmini tepe bellek: Aşama 1 ve CE eğitiminde 15–20 GB. Statik durum ≈9 GB (fp32 ağırlık + gradyan + AdamW). Gerçek
değer ilk çalıştırmadan sonra `hucre_sureleri` içindedir: §7.2 (`bi_dir = …`) ve §9 (`def ce_pairs…`) hücrelerinin
`gpu_tepe_GB`'si (`torch.cuda.max_memory_allocated`, hücre başına).

| Düğme | Etkisi | Sonuçları değiştirir mi |
|---|---|---|
| `BI_MINI_BATCH` 16 → 8 | Yalnız CachedMNRL parça boyu | Hayır (gradyan aynı) |
| `BI_BATCH` 64 → 32 | Batch içi negatif sayısı, adım sayısı | Evet |
| `BI_USE_COSENT=False` | Önbelleksiz 128 dizilik CoSENT dalını kaldırır | Evet |
| `CE_BATCH` 8 → 4 | CE aktivasyonu ∝ `CE_BATCH × CE_DOCS_PER_QUERY` | Evet |
| `CE_DOCS_PER_QUERY` 16 → 8 | Aynı | Evet |
| `CE_MINI_BATCH` | Yalnız ileri geçiş parçası; aktivasyon belleğini **azaltmaz** | — |
| `MAX_SEQ_LEN` / `CE_MAX_LEN` 512 → 384 | Kırpılma artar | Evet |
| `GRAD_CKPT` | **True kalmalı**; CE'de kapalıysa muhtemelen > 40 GB | — |

OOM'dan sonra Python durumu yaşar, ama GPU belleği kendiliğinden boşalmaz. IPython hatanın izini saklar
(`sys.last_traceback` / `sys.last_value` / `sys.last_type`, Python 3.12+'da `sys.last_exc`,
`get_ipython().InteractiveTB.tb`, `last_execution_result`). Bu iz hata anındaki çerçeveleri canlı tutar: model,
trainer, gradyanlar, AdamW durumu, aktivasyonlar. §9'da ayrıca `CE` ve `ce_trainer` globaldir. Bu yüzden tek başına
`free_mem()` yetmez. CFG'yi değiştir, sonra yeni bir hücrede şunu çalıştır:

```python
import sys
if "ce_trainer" in globals(): del ce_trainer   # yalnız §9'dan sonra vardır (başka hücrede: o hücrenin trainer/modeli)
if "CE" in globals(): del CE                   # §7.1/§7.2 OOM'unda ikisi de yok, satırlar atlanır. BI'yi SİLME
sys.last_traceback = sys.last_value = sys.last_type = None
if hasattr(sys, "last_exc"): sys.last_exc = None
ip = get_ipython(); ip.InteractiveTB.tb = None; ip.last_execution_result = None
free_mem(); print(f"{torch.cuda.memory_allocated() / 1e9:.2f} GB ayrılmış")
```

§7.1 ve §7.2'de model ve trainer `train_bi_encoder` içinde yereldir; orada yalnız hata izini temizlemek yeter.
Ayrılmış bellek yalnız hâlâ gereken nesnelere inmeli: §9'dan önce yalnız `BI` (BGE-M3 fp32 ağırlıkları, ≈2,3 GB),
§7.2'den önce ≈0. İnmezse çalışma zamanını yeniden başlat; resume olmadığı için her şey §0'dan yeniden çalışır (bu belgenin §5'i).
İndiyse hata veren hücreyi yeniden çalıştır. §7.2 ve §9 hücreleri kendi içinde sıfırdan eğitir.

## 5. Kurtarma: kaldığı yerden devam yok

- Notebook'ta **resume yoktur**. Hiçbir hücre mevcut model veya sonuç olup olmadığına bakmaz; HF checkpoint'leri
  eğitimden hemen sonra silinir.
- Drive'a ilk yazma §13'tedir. Bağlantı ondan önce koparsa her şey kaybolur.
- `/content/outputs/bi_encoder_best` ve `cross_encoder_best` (fp32) yalnız aynı VM yaşadıkça vardır.
  `SentenceTransformer(path)` ve `CrossEncoder(path)` ile yüklenebilirler. Atlamak için hücreleri elle değiştirmek
  gerekir; `best_scale`, `temp_res`, `bi_log_df`, `ce_log_df` gibi değişkenleri de doldurman lazım.
- §13 `drive.flush_and_unmount()` çağırır. Sonraki hücreler Drive'a yazamaz; yeniden yazmak için §2'yi tekrar çalıştır.
- **Karar senin:** Uzun bir deney planlıyorsan önce şu altyapıyı ekleyebilirsin:
  - en iyi modelleri ve günlükleri her aşamadan sonra Drive'a kopyalamak;
  - "varsa yükle, yoksa eğit" bayrakları (örn. `CFG["REUSE_STAGE1"]`);
  - "Aşama 1'den sonra dur" gibi kısmi çalıştırma bayrakları.

  Böylece Aşama 2 deneyleri Aşama 1'i yeniden eğitmeden yapılabilir. Değiştirdiğin her şeyi kuru çalıştırmayla
  doğrula.

## 6. Bölümler arası bağımlılıklar (atlamadan önce)

- §11, sekiz `RESULTS` anahtarının hepsini ister; biri eksikse KeyError verir.
- `S_LEGACY` (B3) şuralarda kullanılır: §6.1, §11.1, §12, §12.1. `S_ZS_SMALL` (B2) §12.1'de kullanılır.
- §10.1 şunları ister: `HYB`, `CAND`, `CE_S`, `S_BM25`, `S_DENSE`, `ltr_tr`, `LTR_BEST`, `n_est`. §7'nin `E_base_val`
  / `E_base_jobs`'u yoksa "Ref" satırı ve "İnce ayar (val)" karşılaştırması sessizce düşer.
- §12.1, `cross_encoder_best`'i diskten okur.
- §13 şunları ister: `temp_res`, `study`, `best_scale`, `LTR_BEST`, `n_est`, `ABL`, `COLD_TBL`, `COMP`, `DIAG`,
  `TARGETS`, `bi_log_df`, `ce_log_df`, `VAL_TBL`, `VAL_COMP`, `VAL_ZS_BASE`, `BUNDLE_FP`. §10.1'in üç CSV'sini de
  pakete kopyalar; §10.1 atlanırsa §13 hata verir (`NameError` / `FileNotFoundError`).
- `TUNE_TEMPERATURE=False` güvenle atlar (`best_scale = BI_SCALE`).

## 7. Çalıştırmadan sonra geri getirilecekler

Kullanıcıdan şunları iste ve `runs/<YYYY-MM-DD>_<etiket>/` altına koy (yapı: `runs/README.md`):

| Dosya | Nerede | Neden |
|---|---|---|
| Çalıştırılmış `.ipynb` | Colab: Dosya → İndir → .ipynb | Tüm yazdırılan satırlar, tablolar, uyarılar |
| `run_summary.json` | `MyDrive/cv_ilan_sota/` | Karar: `val_sinyalleri`, `val_zero_shot_base(WARM)`, `val_tam_hat(WARM)`, `val_bilesen_testleri`, `tani`. Ayrıca süreler (`hucre_sureleri`, `sureler_sn`, `toplam_sure_dk`), bellek, `paket_parmak_izi`, sürümler, CFG, test değerlendirme sayısı. `hedefler` / `ana_ablation` yalnız kayıttır |
| `val_pipeline.csv`, `val_component_tests.csv`, `per_query_val.csv` | aynı | **Karar:** §10.1 val tam hat ve bileşen testleri. Sorgu bazlı val (0–1): hata kovaları (CV altın sütunlarıyla kovalamak değerlendirme tarafında serbest), iki çalıştırma arası eşleştirilmiş randomizasyon |
| `stage1_training_log.csv`, `stage2a_training_log.csv` | aynı | Val eğrileri (karar) |
| `per_query_test.csv` | aynı | Yalnız kayıt: final raporun hata analizi. Kaldıraç ya da hedefli veri bundan seçilmez |
| `ablation_test.csv`, `ablation_component_tests.csv`, `cold_start_test.csv` | aynı | Test tabloları, yalnız kayıt |
| `figures/` | aynı | Görseller |

`MyDrive/cv_ilan_sota/` sabit bir klasördür. §13 her çalıştırmada aynı adlarla üzerine yazar ve klasörü hiç temizlemez.
`figures/` birleştirilerek kopyalanır (`copytree(..., dirs_exist_ok=True)`), eski PNG'ler kalır; örneğin
`TUNE_TEMPERATURE=False` iken eski `temperature_sweep.png`. §13'e ulaşmayan bir çalıştırmada klasördeki her şey önceki
çalıştırmaya aittir. Bu yüzden:
- dosyalar bir sonraki çalıştırmadan **önce** indirilir ya da kopyalanır;
- `runs/`'a koymadan önce `run_summary.json` → `paket_parmak_izi`, `cfg` ve `toplam_sure_dk`, çalıştırılmış
  `.ipynb`'deki §2, CFG ve §13 çıktılarıyla aynı olmalı. Değilse dosyalar bu çalıştırmanın değildir (`runs/README.md`).
- §2 `Drive bağlanmadı …` yazdıysa §13 Drive'a hiçbir şey yazmaz: bu çalıştırmanın dosyaları yalnız `/content/outputs/`
  ve `/content/export_model_bundle.zip` içindedir, oradan indirilir; Drive'dakiler önceki çalıştırmanındır;
- önceki dosyalar alındıktan sonra yeni çalıştırmadan önce `MyDrive/cv_ilan_sota/figures/` silinir ve aynı makinede
  `!rm -rf /content/outputs` çalıştırılır.

Model zip'i (≈2,3 GB) repoya **girmez**. Drive'da yalnız son çalıştırmanınki kalır (aynı adla üzerine yazılır).

## 8. Çıkarım tarafı (paket)

- `export_model_bundle/` içeriği:
  - `bi_encoder/`, `cross_encoder/` (fp16)
  - `lambdamart.txt`, `attribute_models.joblib`, `title_sim.joblib`
  - `jobs.parquet`, `clean_cvs.parquet`, `job_embeddings.npy`, `faiss.index` (sorgulanmaz)
  - `config.json`, `metrics.json`, `run_summary.json`
  - `per_query_test.csv`, `per_query_val.csv`, `val_pipeline.csv`, `val_component_tests.csv`, `ablation_test.csv`,
    `ablation_component_tests.csv`, `cold_start_test.csv`, `figures/`
  - `match_candidate_sota.py`, `README_INFERENCE.md`
- **Kullanım:**
  - CLI: `python match_candidate_sota.py --bundle . --cv-id 20951 [--top-k 10] [--json]`. `--json` ile `Aday …` bilgi
    satırı stderr'e gider, stdout saf JSON'dur.
  - Dosyadan: `--cv-file aday.txt --headline …`
  - API: `--serve` (`POST /match`, `GET /health`)
- **Uyarılar:**
  - joblib dosyaları `match_candidate_sota` modül adına ve eğitimdeki scikit-learn sürümüne bağlıdır
    (`config.json → sklearn_version`).
  - `config.json` içindeki `feature_columns` çekirdekteki `FEATURE_COLUMNS` ile aynı olmalıdır; değilse
    `ValueError`. Yeni öznitelik eklersen eğitim ve çıkarım aynı çekirdeği kullanmalı.
  - Çıkarım yalnız `EXPORT_CFG`'ye (`config.json`) yazılan ayarları bilir (`rrf_k`, `n_candidates`, `mnrl_scale`,
    `ltr_params` …). `SOTAMatcher.match` RRF'yi ağırlıksız çağırır. RRF ağırlığı gibi çıkarımı etkileyen bir ayarı
    değiştirirsen onu `EXPORT_CFG`'ye ve `SOTAMatcher.match`'e de ekle (çekirdek değişikliği: §1'deki `05` adımları).
    Yoksa paket notebook'tan farklı sıralar ve §13+ paritesi düşer.
