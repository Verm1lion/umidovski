# Deney günlüğü

Her deney bir kayıttır. **Hipotez çalıştırmadan önce yazılır**, sonuç sonra eklenir. Kayıtlar silinmez; yanlışsa
altına düzeltme notu düşülür. "Karar" satırı yalnız **val, WARM havuz** sinyaline dayanır (şablondaki "Karar
sinyali"; `03` §A). Notebook'un yazdırdığı bütün test satırları ve tabloları (§4–§12 ve §14; ör. B1/B2/B3, §8 Recall, §11 ABL/COMP, §11.1, §12), `per_query_test.csv` ve `run_summary.json` → `hedefler` /
`ana_ablation` yalnız kayıttır. §10.1 val tabloları hafifçe iyimserdir (val erken durdurma, checkpoint ve LTR
hiperparametre seçiminde kullanıldı): değişiklikleri birbiriyle kıyaslamak içindir, yansız kestirim değildir.
Çıktılar `runs/<tarih>_<etiket>/` altına konur. Drive'daki `MyDrive/cv_ilan_sota/` her çalıştırmada üzerine yazılır:
dosyaları bir sonraki çalıştırmadan **önce** kopyala ve bu çalıştırmaya ait olduklarını doğrula (`runs/README.md`).

## Şablon

```markdown
### E0NN — <kısa ad>   (<YYYY-MM-DD>, runs/<klasör>)
- **Tek değişiklik:** <CFG anahtarı / kod yeri: eski → yeni>
- **Hipotez ve mekanizma:** <neden işe yaramalı; bir hata kovasına dayanıyorsa val kovası: `per_query_val.csv` + CV
  altın sütunları (yalnız değerlendirme tarafında)>
- **Beklenen val etkisi:** <hangi sinyal, hangi yönde, ≈ne kadar> (ör. Aşama 1 `eval_val_graded_ndcg@10` +0,02;
  log'lar 0–1, yazdırılan tablolar yüzde)
- **Karar sinyali (val, WARM):** <Aşama 1/2a `eval_val_graded_ndcg@10` | `val_zero_shot_base(WARM)` | `ltr_cv_ndcg10` |
  §9.1 `aday_tavani_val(WARM)` | §10.1 `VAL_TBL` / `VAL_COMP` | `per_query_val.csv` | `scripts/13`>
- **Çalıştırma:** `paket_parmak_izi`=<metadata / çekirdek, 12 hane; repodaki `sha256sum` ile aynı>,
  `toplam_sure_dk`=<x>, `gpu_tepe_bellek_GB`=<y>; çekirdek değiştiyse parquet sha256'ları:
  <aynı | değişti → veri değişikliği>
- **Sonuç (val):** <sayı(lar) [%95 GA], dosya yolu>; önceki çalıştırmaya göre ΔNDCG@10 ve eşleştirilmiş p
  (`per_query_val.csv`: aynı `sistem`, `qid` ile eşleştir, `core.paired_randomization_test`)
- **Test (yalnız kayıt):** <S2 NDCG@10 / Top-1 / P@10 [GA]>, test_degerlendirme_sayisi=<n>, toplam tam çalıştırma=<k>
- **Tavanlar:** val (karar): `aday_tavani_val(WARM)` → `NDCG@10`=<x> (kıyas: val WARM BM25 ilk-50 84,14 / ilk-100
  90,67), kâhin P@10 (val, WARM) 83,22; test (yalnız kayıt): ikili kusursuz 91,36, kâhin P@10 83,38
- **Karar:** <tut / geri al / genişlet> — gerekçe (val'e dayalı)
- **Sonraki adım:** <tek cümle>
```

## Kullanıcı kararları kaydı

| Tarih | Soru | Karar | Etkisi |
|---|---|---|---|
| — | P@10 > 85 testte ulaşılamaz (kâhin 83,38). Hedef nasıl ifade edilsin? | *bekliyor* | — |
| — | GPU bütçesi: kaç A100 saati / kaç tam çalıştırma? | *bekliyor* | Deney planlarının üst sınırı; ilk deney planından önce sorulur (`00` §4 madde 6) |

## Kayıtlar

### E000 — Hazırlık durumu ve test kümesiyle temaslar (2026-10-16)

- **Durum:**
  - Veri paketi, notebook (55 hücre) ve çekirdek hazır. CPU kuru çalıştırma OK: `reports/notebook_dogrulama.json` →
    `statik` OK, `dry_run.durum` OK, `zaman` 2026-10-16 18:09:39, `notebook_sha256` güncel `.ipynb` ile aynı.
  - Paket parmak izi (sha256[:12]): `metadata.json` b0eb93042e32, `match_candidate_sota.py` d6a16586ffd7.
  - Notebook val karar sinyallerini WARM havuzda yazdırır: §7 val sıfır-atış (`val_zero_shot_base(WARM)`), §9.1
    `aday_tavani_val(WARM)`, §10.1 `VAL_TBL` / `VAL_COMP`; sorgu bazlı val sonuçlarını `per_query_val.csv`'ye yazar.
    `run_summary.json` ayrıca `hucre_sureleri` ve `paket_parmak_izi` tutar.
  - GPU'da hiç çalıştırma yok, nöral sistemlerin sonucu yok; süre ve bellek ölçülmedi.
- **GPU'suz referanslar:**
  - Val (karar ölçeği, WARM): leksikal kaldıraçlar, `reports/lexical_lever_probe_val.json` (`scripts/13`).
  - Test (yalnız kayıt ve sağlama): BM25 NDCG@10 53,66 [51,5; 55,8], Top-1 64,68, P@10 38,37
    (`reports/local_bm25_test.json`; notebook B1 bunu birebir vermeli). BM25 + LambdaMART (leksikal) 66,05
    [63,9; 68,2], Top-1 72,85 (`reports/local_lexical_ltr.json`).
- **Tavanlar** (`reports/metric_ceilings.json`):
  - Val, WARM (karar ölçeği): kâhin P@10 83,22; BM25 ilk-50 / ilk-100 / ilk-200 → kâhin NDCG@10 84,14 / 90,67 / 94,58.
  - Tam havuz (karar için kullanılmaz; test yalnız ulaşılabilirlik raporu için): kâhin P@10 test 83,38 / val 85,08;
    ikili kusursuz NDCG@10 test 91,36 / val 91,96 (WARM sürümü yok); BM25 ilk-50 → kâhin test 86,95 / val 83,15.
- **Test kümesiyle temaslar (dürüst sayım):**
  1. Faz 2 raporu için BM25 ve leksikal LTR testte ölçüldü (yukarıdaki iki dosya). LTR bir kez değerlendirildi
     (`scripts/11`). BM25 deterministiktir; `scripts/10`, `11` ve `12` aynı sayıyı verir.
  2. Kâhin ve ikili kusursuz tavanları yalnız test etiketlerinin yapısıdır; model ve seçim yoktur, tanımın
     özellikleridir. BM25 ilk-K → kâhin tavanları ise BM25'in test aday kümesine de dayanır: `scripts/12` testte
     BM25'i çalıştırır (tam ve WARM havuz). Eğitim ve seçim yoktur; `bm25_dogrudan` = 53,66, 1. maddedeki ölçümün
     aynısıdır. Karar için yalnız val **WARM** karşılıkları kullanılır (kâhin NDCG@10 84,14 / 90,67 / 94,58); tam
     havuz val değerleri (83,15 / 89,71 / 93,95) de karar için değildir.
  3. Devir hazırlığında bir analiz ajanı, leksikal hatta ≈14 LTR varyantını (ISCO-4 özniteliği, kNN, K=50/100,
     train sorgularıyla çapraz uydurma) ve BM25 Top-1 hata kategorilerini **test üzerinde** keşif amaçlı ölçtü.
     - Bu sayılar repoda değildir; hiçbir kararda ve hiçbir belgede kanıt ya da yönlendirme olarak kullanılmaz
       (kaldıraç sırası, kanıt sütunları, hata kovaları dahil).
     - Yerine yalnız val kullanan `scripts/13_lexical_lever_probe.py` yazıldı: WARM havuz (cold ilanlar ne adayda ne
       qrels'te), 5 katlı GroupKFold, kat dışı sıralama (`reports/lexical_lever_probe_val.json`). Train sorgularıyla
       çapraz uydurmanın ve Top-1 hata kategorilerinin val ölçümü repoda yok; gerekiyorsa val'de ölç ve dosyaya yaz.
     - **Sonuç:** Leksikal varyantlar açısından test kümesi artık tamamen "bakir" değil. Nöral hat testte hiç ölçülmedi.
       Savunmada bu açıkça söylenmeli.
- **Karar:** İlk A100 çalıştırması varsayılan `CFG` ile yapılır (E001).

### E001 — İlk A100 çalıştırması, varsayılan CFG   (planlandı)

- **Tek değişiklik:** yok; referans çalıştırma.
- **Hipotez:**
  - Sağlık satırları `02` §3 ile uyumlu olacak. §2 `paket parmak izi` = {`metadata.json`: b0eb93042e32,
    `match_candidate_sota.py`: d6a16586ffd7}, yani repodaki
    `sha256sum data_bundle/metadata.json match_candidate_sota.py | cut -c1-12` (taze klonda önce
    `unzip -o data_bundle.zip`). B1 birebir 53,66 (test; yalnız sağlama, BM25 deterministiktir).
  - İnce ayarlı dense val'de sıfır-atış BGE-M3'ü geçecek: en iyi `eval_val_graded_ndcg@10` (`val_sinyalleri` →
    `bi_val_en_iyi`, 0–1) × 100 > §7 `Zero-shot base bi-encoder (VAL, WARM havuz — …)` NDCG@10
    (`val_zero_shot_base(WARM)`, yüzde); §10.1 "İnce ayar (val)" ΔNDCG@10 > 0. §7'nin ilk `Zero-shot base
    bi-encoder:` satırı testtir, bu kıyasta kullanılmaz.
  - §9.1 `aday_tavani_val(WARM)` NDCG@10, val WARM BM25 ilk-50 tavanının (84,14) belirgin üstünde olacak (hücre WARM
    tabanları `Kıyas (val, WARM, kâhin NDCG@10): …` satırında yazdırır).
- **Ölçülecekler:**
  - dosyaların bu çalıştırmaya ait olduğu (`paket_parmak_izi`, `cfg`, `toplam_sure_dk` ↔ `.ipynb`'nin §2, CFG ve §13
    çıktıları; `runs/README.md`);
  - süre ve tepe bellek: `toplam_sure_dk`, `hucre_sureleri` (CFG hücresinden sonraki her hücre: ilk satır, sn, GPU
    tepe GB), `gpu_tepe_bellek_GB` (hücre tepelerinin en büyüğü); `sureler_sn` yalnız eğitim koşularını tutar. §10
    Optuna süresi hiç ölçülmedi (`LTR_TRIALS` × `LTR_FOLDS` = 40 × 5 = 200 LightGBM eğitimi, CPU); ilk değer
    `hucre_sureleri`'nde `hucre_ilk_satir` = `from sklearn.model_selection import GroupKFold` olan §10 hücresinin
    `sn`'sidir (Optuna + son LambdaMART eğitimi + test puanlama + SHAP; Optuna için üst sınır);
  - kırpılma oranları (`tani` → `kirpilma_*`);
  - §9.1 `aday_tavani_val(WARM)`;
  - Aşama 1 ve CE val eğrileri (`stage1_training_log.csv`, `stage2a_training_log.csv`) ve `val_zero_shot_base(WARM)`;
  - val tam hat ve bileşen testleri: §10.1 `VAL_TBL` / `VAL_COMP` (`val_pipeline.csv`, `val_component_tests.csv`);
    `per_query_val.csv` sonraki çalıştırmalarla eşleştirilmiş kıyas için saklanır;
  - hedef satırları ve test bileşen testleri (§11; yalnız kayıt), `test_degerlendirme_sayisi`.
- **Karar sinyali:** §9.1 `aday_tavani_val(WARM)` NDCG@10 (`04` §1 düğüm 3):
  - < 92 → Aşama 2 işine (CE, LTR) başlama; önce Aşama 1 / `N_CANDIDATES` / RRF (RRF ağırlığı değişirse
    `EXPORT_CFG` ve `SOTAMatcher.match` de değişir: çekirdek değişikliği);
  - 92–95 → gri bölge: Aşama 1 / aday iyileştirmesi muhtemelen hâlâ en verimlisi;
  - ≥ 95 → Aşama 2 işi karşılık verebilir.

  Kıyas: val WARM BM25 ilk-50 84,14 / ilk-100 90,67. Val aday tavanları test tavanlarından birkaç puan düşüktür; test
  tavanı (`aday_tavani_test(tam_havuz,yalniz_rapor)`) yalnız rapor içindir.
