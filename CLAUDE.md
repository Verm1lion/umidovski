# CV ↔ İş İlanı Eşleştirme — ajan talimatları

Bu repo CV → ilan sıralama sisteminin sızıntısız yeniden inşasıdır. Sistem iki aşamalıdır: BGE-M3 + BM25 (RRF),
ardından bge-reranker-v2-m3 + LambdaMART. Eğitim Colab A100 notebook'unda yapılır. **İşe başlamadan önce
`ajan/00_BASLA_BURADAN.md`'yi baştan sona oku**, sonra `ajan/01_TARIF.md`'yi.

## Değişmez kurallar

1. **Test mühürlü zarftır.** Hiçbir karar test sayısına göre verilmez: τ, epoch, checkpoint, öznitelik, model ya
   da hangi çalıştırmanın raporlanacağı. Kararlar val sinyalleriyle verilir (`ajan/03_KONTROL_LISTELERI.md` §A).
2. **Sızıntı kapısı gevşetilmez.** Kapı `scripts/05_build_dataset.py` → `leakage_checks()` içindedir. Cold ilanlar
   hiçbir eğitim veya seçim sinyaline girmez. `gold_*` ve `legacy_*` sütunları öznitelik yapılmaz. Sentetik veri
   yalnız train'dedir ve val/test metninden üretilmez. Vitrin CV'leri testte kalır.
3. **Metrik tanımı ve alaka kuralları kullanıcının kararıdır.** `REL_THRESHOLD`, `grade_matrix()`, P@10 hesabı ve
   hedeflerin yeniden ifadesi bu kapsamdadır. P@10 > 85 testte yapısal olarak ulaşılamaz (kâhin 83,38). Bunu
   raporla; tanımla oynayarak "tutturma".
4. **Notebook elle düzenlenmez.** Kaynak `scripts/07_make_notebook.py`'dir. Her değişiklikten sonra:
   ```bash
   python scripts/07_make_notebook.py
   python scripts/08_dry_run_notebook.py --bundle data_bundle.zip --workdir work/dry_run --kernel <kernel>
   ```
   `reports/notebook_dogrulama.json` içinde `"durum": "OK"` görmeden iş bitmiş sayılmaz.
5. **Çekirdek paketin içindedir.** `match_candidate_sota.py` değişince şunları yap:
   - `python scripts/05_build_dataset.py` (paket + zip + kapı);
   - kontrol: `unzip -p data_bundle.zip data_bundle/match_candidate_sota.py | cmp - match_candidate_sota.py`.

   Colab'da eski `/content/data_bundle` silinmezse yeni zip açılmaz.
6. **Her deney günlüğe yazılır.** Hipotez çalıştırmadan **önce**, sonuç sonra yazılır (`ajan/06_DENEY_GUNLUGU.md`).
   Çalıştırma çıktıları `runs/<tarih>_<etiket>/` altına konur (`runs/README.md`).
7. **Kanıtsız iddia yok.** Her sayı bir dosyaya (JSON/CSV) dayanır. NDCG için %95 GA, karşılaştırmalar için
   eşleştirilmiş randomizasyon p-değeri verilir. Sonuçlar tavanlarıyla birlikte yazılır
   (`reports/metric_ceilings.json`).

## Ortam

- Python ≥ 3.10. Notebook'un sabit sürümleri: sentence-transformers 6.1.0, transformers 5.17.0, accelerate 1.15.0,
  datasets 5.1.0, lightgbm 4.7.0, optuna 5.0.0, rank_bm25 0.2.2, faiss-cpu 1.15.1. Ayrıca torch, scikit-learn,
  pandas, pyarrow, nbformat, nbclient, ipykernel, pyflakes gerekir.
- GPU yoksa yerel olarak yalnız şunlar çalışır: kuru çalıştırma (küçük rastgele modeller, `CV_DRY_RUN=1`) ve
  leksikal betikler (`scripts/11`, `12`, `13`).
- Colab'ı kullanıcı çalıştırır. Döngü `ajan/00_BASLA_BURADAN.md` §5'te anlatılıyor.

## Hızlı başvuru

| Ne | Nerede |
|---|---|
| Tavanlar (kâhin, ikili kusursuz, aday kümesi) | `reports/metric_ceilings.json` (`scripts/12_metric_ceilings.py`) |
| Kaldıraçların val ölçümü (leksikal, GPU'suz) | `reports/lexical_lever_probe_val.json` (`scripts/13_lexical_lever_probe.py`) |
| BM25 / leksikal LTR test referansı | `reports/local_bm25_test.json`, `reports/local_lexical_ltr.json` |
| Veri ve qrels raporu | `reports/FAZ2_VERI_RAPORU.md` (`scripts/10` üretir) |
| Kurulum, teori, sınırlılıklar | `AKADEMIK_SAVUNMA_VE_KURULUM_REHBERI.md` |
| Sağlık satırları ve beklenen değerler | `ajan/02_CALISTIRMA_VE_IZLEME.md` §3 |
| Hedef tutmazsa | `ajan/04_HEDEF_TUTMAZSA.md` |

Kullanıcıya görünen metinler (belgeler, rapor, notebook) Türkçedir; ondalık ayırıcı virgüldür.
