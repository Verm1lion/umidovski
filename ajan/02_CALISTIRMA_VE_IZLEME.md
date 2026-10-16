# Çalıştırma ve izleme (runbook)

Kullanıcı için adım adım Colab tarifi: `AKADEMIK_SAVUNMA_VE_KURULUM_REHBERI.md` BÖLÜM 1. Bu belge **ajan** içindir:
- her bölümün ne yaptığı;
- hangi satırın sağlık göstergesi olduğu;
- sorun çıkınca hangi düğmenin çevrileceği;
- çalıştırmadan sonra neyin geri getirileceği.

Bütün süre ve bellek değerleri **tahmindir**. Notebook GPU'da hiç çalışmadı. İlk gerçek çalıştırmanın
`run_summary.json` dosyası (`sureler_sn`, `gpu_tepe_bellek_GB`) bu tahminlerin yerini almalı.

## 1. Çalıştırmadan önce (ajan)

```bash
python scripts/07_make_notebook.py                     # notebook'u üret (elle düzenleme yok)
python scripts/08_dry_run_notebook.py --bundle data_bundle.zip --workdir work/dry_run --kernel <kernel>
cat reports/notebook_dogrulama.json                     # "durum": "OK", RUN_OK true, export_zip true olmalı
```

Çekirdek veya veri değiştiyse önce `python scripts/05_build_dataset.py` çalıştır. Bu betik paketi, zip'i ve sızıntı
kapısını yeniden üretir. Kapı geçmezse paket yazılmaz.

Kullanıcıya şunları söyle:
- hangi dosyaları yükleyeceğini (`.ipynb`, değiştiyse `data_bundle.zip`);
- aynı Colab makinesini kullanıyorsa ilk hücreden önce şunu çalıştırmasını: `!rm -rf /content/data_bundle /content/outputs`.

## 2. Bölüm haritası

| § | Ne yapar | Ana CFG | Yazdığı dosyalar | Tahmini süre (A100-40GB) |
|---|---|---|---|---|
| 0 | `pip install` (sabit sürümler), GPU bayrakları, sürüm uyarısı | — | — | 1–3 dk |
| 1 | `CFG` | hepsi | — | — |
| 2 | Drive bağla, `data_bundle.zip` bul/aç, sızıntı assert'leri | `USE_DRIVE` | — | <1 dk |
| 3 | Metrik altyapısı (`evaluate`, `fmt`, `EVAL_LOG`) | `REL_THRESHOLD` | — | — |
| 4 | B1 BM25 (+ rank_bm25 paritesi) | — | — | <1 dk |
| 5 | B2 sıfır-atış e5-small (ilk HF indirmesi) | `BASELINE_SMALL` | — | 1–2 dk |
| 6 | B3 eski yaklaşım (e5-small + CosineSimilarityLoss); §6.1 sızıntı gösterimi | `LEGACY_*` | `figures/b3_training.png` | 4–10 dk |
| 7 | Aşama 1 hazırlık: BGE-M3 sıfır-atış, dense negatif madenciliği, MNRL/CoSENT satırları, yanlış-negatif tanısı | `DENSE_MINED_NEG`, `COSENT_PER_GRADE` | — | 1–2 dk |
| 7.1 | τ taraması (yarım veri, 1 epoch, CoSENT yok) | `TUNE_TEMPERATURE`, `TEMP_GRID`, `TEMP_FRACTION` | `figures/temperature_sweep.png` | 5–10 dk |
| 7.2 | Aşama 1 tam eğitim + erken durdurma | `BI_*`, `EVALS_PER_EPOCH`, `EARLY_STOP_PATIENCE` | `stage1_training_log.csv`, `bi_encoder_best/` | 18–35 dk |
| 8 | Hibrit RRF (dense ⊕ BM25) | `RRF_K` | — | <1 dk |
| 9 | Aşama 2a cross-encoder (LambdaLoss) | `CE_*`, `N_CANDIDATES` | `stage2a_training_log.csv`, `cross_encoder_best/` | 22–35 dk |
| 9.1 | **Tanı:** aday kümesi tavanı (val), belirteç kırpılma oranları, GPU tepe bellek | — | (`run_summary.json` → `tani`) | <1 dk |
| 10 | Öznitelik sınıflandırıcıları, LTR çerçeveleri, LambdaMART (Optuna 40 deneme × 5 kat) | `LTR_*` | `figures/ltr_importance.png` | 5–20 dk (CPU) |
| 11 | Ablation (5 ana + 3 ek sistem), bileşen testleri, HEDEF karşılaştırması | — | `ablation_test.csv`, `ablation_component_tests.csv` | <1 dk |
| 11.1 | Sektör tablosu, soğuk-başlangıç (301 cold ilan) | — | `cold_start_test.csv` | 2–3 dk |
| 12 | Vitrin hata analizi, dikkat + SHAP, Top-1 hataları | — | `figures/attention_*`, `shap_*` | 1–2 dk |
| 13 | Dışa aktarım, `run_summary.json`, `per_query_test.csv`, zip, Drive kopyası, `flush_and_unmount` | `EXPORT_HALF` | `export_model_bundle.zip` (≈2,3 GB) | 2–6 dk |
| 13+ | Paket ↔ notebook paritesi, FastAPI duman testi | — | — | 1–2 dk |
| 14 | Özet, `RUN_OK.json` | — | `RUN_OK.json` | — |

Toplam tahmin 65–125 dakikadır; notebook başlığı "≈ 1,5–2,5 saat" der.

Optuna için yerel bir CPU ölçümü var: 4 CPU'da deneme başına ≈25 sn sürdü, yani 40 deneme ≈17 dakika. Rehberdeki
"3–5 dk" tahmini muhtemelen düşüktür.

## 3. Sağlık satırları (sırayla oku)

| Nerede | Beklenen | Değilse |
|---|---|---|
| §0 | `GPU: NVIDIA A100… | bf16=True | tf32=True | fp16=False` | A100 değilse batch'leri düşür (§4) |
| §0 | `UYARI: Sabitlenmiş sürümlerden farklı…` **yok** | Oturumu yeniden başlat, tekrar çalıştır |
| §2 | `Çekirdek modül sürümü: 1.0.0`, `Sızıntı koruyucuları: TAMAM` | Paket bozuk ya da eski; dur |
| §2 | `İlan havuzu: {'warm': 2699, 'cold': 301} \| qrels satırı: 608691` | Veri farklı; dur |
| §2 | `train anchor: 2786 \| val sorgu: 457 \| test sorgu: 461 (… 8)` | `USE_SYNTHETIC=False` ise anchor 2076 |
| §4 | `BM25 paritesi (rank_bm25 ile): TAMAM` ve B1 = **53.66 / 74.03 / 38.37 / 48.53 / 26.59 / 47.46 / 64.68 / 9.49** | Veri/çekirdek uyuşmazlığı; dur |
| §7 | `MNRL satırı: 4883 … CoSENT çifti: 9719` | Satır üreticisi değişmiş |
| §7 | Yanlış-negatif tanısı (üç satır + toplam) | Toplam yüksekse (≳%3) yanlış-negatif maskesi düşün (`04`) |
| §7.1 | `ölçek=20.0 (τ=0.050) -> val NDCG@10=…`, `Seçilen ölçek: …` | İki değer çok yakınsa seçim gürültüdür |
| §7.2 | `stage1_training_log.csv` → `eval_val_graded_ndcg@10` yükseliyor; en iyi değer "Ref: Zero-shot base" val'inden yüksek | İlk değerlendirme en iyiyse LR/epoch; hiç yükselmiyorsa hata ara (`01` §3) |
| §8 | `Recall@10/50/100 (aday tavanı)` | R@50 test kâhini 91,37 (çok alakalılı sorgular yüzünden 100 olamaz) |
| §9 | `CE eğitim listesi: 2786 \| ortalama liste uzunluğu: 16.0` | — |
| §9.1 | `Aday kümesi tavanı (val, …)` → NDCG@10 | **< 92 ise Aşama 2 hedefe götüremez**; önce Aşama 1 / `N_CANDIDATES` |
| §9.1 | Belirteç kırpılma oranları (BI ilan/CV, CE çift) | Yüksekse `MAX_SEQ_LEN` / `CE_MAX_LEN` ya da sıkıştırma (`04`) |
| §10 | `Val sektör sınıflandırıcı doğruluğu: %≈80.5 \| yaka: %≈94.3` | Belirgin düşükse sklearn sürümü / veri farkı |
| §10 | `LTR çerçeveleri: {'val': (22850, 38), 'test': (23050, 38)} \| özellik sayısı: 35` | `N_CANDIDATES` değiştiyse satırlar ona göre |
| §10 | `En iyi CV NDCG@10 (val, GroupKFold…)` | Test NDCG ile büyük fark: LTR aşırı uyumu veya val→test kayması |
| §11 | Ablation tablosunda `sorgu = 453` her satırda | Farklıysa değerlendirme kümesi değişmiş |
| §11 | `Hedef karşılaştırması` satırları | Bkz. `04`. P@10 için tavan 83,38 |
| §11.1 | Soğuk-başlangıç tablosu `sorgu = 360` | — |
| §13 | `run_summary.json yazıldı \| test değerlendirme sayısı: N` | — |
| §13 | `Paket: … (≈2.3 GB)`, `Drive yedeği senkronize edildi` | Drive kotası dolu olabilir; kopyalama try içinde değil |
| §13+ | `Paket ↔ notebook paritesi (30 test sorgusu): … örtüşme ≈ %100, Top-1 aynı ≈ %100` | <%95 ise hassasiyet kayması (`05` §A) |
| §13+ | `FastAPI POST /match -> 200` | fastapi kurulu değilse atlanır (sorun değil) |

## 4. Bellek yetmezse (OOM)

Tahmini tepe bellek: Aşama 1 ve CE eğitiminde 15–20 GB. Statik durum ≈9 GB (fp32 ağırlık + gradyan + AdamW).

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

OOM'dan sonra Python durumu yaşar. CFG'yi değiştir, `free_mem()` çağır, hata veren hücreyi yeniden çalıştır.
§7.2 ve §9 hücreleri kendi içinde sıfırdan eğitir. Çekirdek ölürse her şey baştan çalışır.

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
- §12.1, `cross_encoder_best`'i diskten okur.
- §13 şunları ister: `temp_res`, `study`, `best_scale`, `LTR_BEST`, `n_est`, `ABL`, `COLD_TBL`, `COMP`, `DIAG`, `TARGETS`.
- `TUNE_TEMPERATURE=False` güvenle atlar (`best_scale = BI_SCALE`).

## 7. Çalıştırmadan sonra geri getirilecekler

Kullanıcıdan şunları iste ve `runs/<YYYY-MM-DD>_<etiket>/` altına koy (yapı: `runs/README.md`):

| Dosya | Nerede | Neden |
|---|---|---|
| Çalıştırılmış `.ipynb` | Colab: Dosya → İndir → .ipynb | Tüm yazdırılan satırlar, tablolar, uyarılar |
| `run_summary.json` | `MyDrive/cv_ilan_sota/` | Hedef kararı, val sinyalleri, tanılar, süreler, bellek, sürümler, CFG, test değerlendirme sayısı |
| `per_query_test.csv` | aynı | Sorgu bazlı hata analizi (sistem × sorgu) |
| `stage1_training_log.csv`, `stage2a_training_log.csv` | aynı | Eğriler |
| `ablation_test.csv`, `ablation_component_tests.csv`, `cold_start_test.csv` | aynı | Sonuç tabloları |
| `figures/` | aynı | Görseller |

Model zip'i (≈2,3 GB) repoya **girmez**; yalnız Drive'da kalır.

## 8. Çıkarım tarafı (paket)

- `export_model_bundle/` içeriği:
  - `bi_encoder/`, `cross_encoder/` (fp16)
  - `lambdamart.txt`, `attribute_models.joblib`, `title_sim.joblib`
  - `jobs.parquet`, `clean_cvs.parquet`, `job_embeddings.npy`, `faiss.index` (sorgulanmaz)
  - `config.json`, `metrics.json`, `run_summary.json`
  - `match_candidate_sota.py`, `README_INFERENCE.md`
- **Kullanım:**
  - CLI: `python match_candidate_sota.py --bundle . --cv-id 20951 [--top-k 10] [--json]`
  - Dosyadan: `--cv-file aday.txt --headline …`
  - API: `--serve` (`POST /match`, `GET /health`)
- **Uyarılar:**
  - joblib dosyaları `match_candidate_sota` modül adına ve eğitimdeki scikit-learn sürümüne bağlıdır
    (`config.json → sklearn_version`).
  - `config.json` içindeki `feature_columns` çekirdekteki `FEATURE_COLUMNS` ile aynı olmalıdır; değilse
    `ValueError`. Yeni öznitelik eklersen eğitim ve çıkarım aynı çekirdeği kullanmalı.
