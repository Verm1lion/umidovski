# Kontrol listeleri

Kısa tutuldu, çünkü her seferinde okunacaklar. İşaretlemeden geçme.

## A. Bir değişikliğe başlamadan önce

- [ ] Hipotez `06_DENEY_GUNLUGU.md`'ye yazıldı: ne değişiyor, hangi val sinyali hangi yönde ne kadar oynamalı,
      ve neden (mekanizma).
- [ ] Değişiklik **tek** bir şey. Birden çok şey değişiyorsa ayrı deneylere bölündü.
- [ ] Karar hakkı kontrolü yapıldı (`00` §4). Metrik tanımı, alaka kuralı, dış LLM API'si, yeni sentetik veri ya da
      LLM ile yeniden anotasyon (maliyet), lisans, ürün davranışını değiştiren yanlılık veya GPU bütçesi işin
      içindeyse önce kullanıcıya soruldu. `gold_*` / `legacy_*` sütunlarını (ilan tarafı dahil) öznitelik yapmak
      soru değildir: **asla** yapılmaz (§F).
- [ ] Kararın dayandığı sinyal val'den geliyor ve **WARM havuzda** ölçülüyor (cold ilanlar ne adayda ne qrels'te):
      Aşama 1/2a log'ları (`eval_val_*`), `val_zero_shot_base(WARM)`, `ltr_cv_ndcg10`, §9.1
      `aday_tavani_val(WARM)`, §10.1 `VAL_TBL` / `VAL_COMP` / `per_query_val.csv`, `scripts/13` (val GroupKFold, kat
      dışı). §10.1 hafifçe iyimserdir (val erken durdurma, checkpoint ve LTR hiperparametre seçiminde kullanıldı):
      değişiklikleri birbiriyle kıyaslamak içindir, yansız kestirim değildir.
- [ ] Testten gelmiyor. Notebook'un yazdırdığı bütün test satırları ve tabloları (§4–§12 ve §14; ör. B1/B2/B3, §8 Recall, §11 ABL/COMP, §11.1, §12),
      `per_query_test.csv` ve `run_summary.json` → `hedefler` / `ana_ablation` yalnız kayıttır. Hazırlıkta testte
      yapılan keşif ölçümleri (leksikal LTR varyantları, BM25 Top-1 hata kategorileri) repoda yoktur ve hiçbir kaldıraç
      seçiminde kanıt sayılmaz (`06` E000).

## B. Çalıştırmadan önce (kod değiştiyse)

- [ ] Notebook değişikliği `scripts/07_make_notebook.py` içinde yapıldı. `.ipynb` dosyası
      `python scripts/07_make_notebook.py` ile üretildi.
- [ ] `match_candidate_sota.py` değiştiyse `python scripts/05_build_dataset.py` çalıştırıldı. Sızıntı kapısı geçti
      ve `data_bundle.zip` yenilendi. Zip'teki çekirdek repo çekirdeğiyle aynı:
      `unzip -p data_bundle.zip data_bundle/match_candidate_sota.py | cmp - match_candidate_sota.py`.
- [ ] **Veri değişmedi mi?** `05` çekirdeği yalnız pakete kopyalamaz, veriyi de onunla üretir: `tr_ascii` / `tr_lower`
      (`spec_key`, unvan maskeli TF-IDF → bölme), `norm_city` (`gold_city` → grade 3 `loc_ok` → qrels),
      `parse_cv_sections` / `parse_seniority_text` / `parse_years_experience` (+ `norm_city`: `text_city`,
      `text_seniority`, `text_years_exp`), `build_cv_struct` / `build_job_struct` (`cv_struct`, `job_struct`,
      `query_text`), `tokenize` / `SparseBM25` / `TitleSimilarity` (triplet negatifleri). Çekirdek değişikliği
      parquet'leri değiştirebilir:
  - `05`'ten **önce** `data_bundle/metadata.json`'u scratch'e yedekle (taze klonda `data_bundle/` yoktur, git'te
    izlenmez: önce `unzip -o data_bundle.zip`). `05` klasörü silip yeniden yazar, `reports/faz2_veri_ozeti.json`'u da
    ezer.
  - Sonra `metadata.json` → `dosyalar` altındaki sha256'ları yedekle karşılaştır; yalnız `match_candidate_sota.py`
    değişmiş olmalı. Bir parquet'in hash'i farklıysa (pyarrow/pandas sürümü de baytları değiştirebilir) içeriği
    pandas ile karşılaştır.
  - `qrels_ground_truth.parquet` ya da `clean_cvs.parquet` / `*_queries.parquet` içindeki `cv_id` / `split`
    değiştiyse **dur ve kullanıcıya sor** (değişmez kural 5; alaka kuralı ya da test kümesi değişir, B1 ve önceki
    çalıştırmalar geçersizleşir). Yalnız `cv_struct` / `job_struct` (`clean_jobs.parquet`) / `query_text` / `text_*`
    sütunları / triplet değiştiyse bu bir veri değişikliğidir; günlüğe yazıldı.
- [ ] Çıkarımı etkileyen bir ayar (ör. RRF ağırlıkları) değiştiyse `EXPORT_CFG`'ye ve `SOTAMatcher.match`'e de girdi
      (çekirdek değişikliği; bugün paket RRF'den yalnız `rrf_k`'yı taşır, ağırlık taşımaz: `SOTAMatcher.match`
      `rrf_scores`'u ağırlıksız çağırır). Yoksa dışa aktarılan paket notebook'tan farklı sıralar; §13+ paritesi düşer.
- [ ] `python scripts/08_dry_run_notebook.py --bundle data_bundle.zip --workdir work/dry_run --kernel <k>` → çıkış
      kodu 0; `reports/notebook_dogrulama.json`: `"statik": "OK"`, `dry_run.durum = "OK"`, `RUN_OK: true`,
      `export_zip: true`, `ast_hatalari` ve `pyflakes` boş; `zaman` bu çalıştırmanın saati; `notebook_sha256` =
      `sha256sum CV_Ilan_SOTA_Egitim_ve_Test_A100.ipynb`. Rapor her çıkış yolunda yeniden yazılır: `HATA`,
      `YAPILMADI`, `ATLANDI (--static-only)` ya da `CALISIYOR`'da kalmış bir rapor başarı değildir.
- [ ] Yeni öznitelik eklendiyse (çekirdek değişikliği: `05` ve "Veri değişmedi mi?" maddeleri de geçerli):
  - çekirdekte `FEATURE_COLUMNS` güncellendi;
  - öznitelik `build_pair_features` içinde hesaplanıyor;
  - `SOTAMatcher` aynı özniteliği çıkarımda da üretiyor;
  - yasaklı ad kullanılmadı (`gold_*`, `legacy_*`, `grade`, `label`, …).
- [ ] Kullanıcıya şunlar söylendi (ayrıntı `02` §1):
  - hangi dosyalar yüklenecek (`.ipynb`; değiştiyse `data_bundle.zip`) ve çalıştırmanın amacı (tek cümle);
  - `MyDrive/cv_ilan_sota/` her çalıştırmada üzerine yazılır: önceki çalıştırmanın dosyaları yeni çalıştırmadan
    **önce** indirilir ya da kopyalanır;
  - yeni zip §2'den önce Dosyalar paneliyle `/content/data_bundle.zip` olarak (ya da `MyDrive/data_bundle.zip`)
    konur; §2 en yeni zip'i seçer ve açılmış paketten farklıysa yeniden açar (`00` §7). Aynı Colab makinesinde
    `!rm -rf /content/outputs`; önceki dosyalar indirildikten sonra `MyDrive/cv_ilan_sota/figures/` silinir (§13
    görselleri birleştirerek kopyalar, eski PNG'ler kalır; `runs/README.md`);
  - §2'deki `paket parmak izi`, `sha256sum data_bundle/metadata.json match_candidate_sota.py | cut -c1-12` ile aynı
    olmalı (şu anki paket: `b0eb93042e32` / `d6a16586ffd7`).

## C. Çalışma sırasında / hemen sonra

`02` §3'teki sağlık tablosunu satır satır geç. Özellikle şunlar:
- [ ] §2: `Açılıyor: …` ya da `Açılmış paket zip ile aynı: …`; `paket parmak izi` beklenen değerle aynı (§B).
      `UYARI: zip bulunamadı; …` görünüyorsa önceden açılmış klasör kullanıldı: parmak izi tutmuyorsa dur.
      `Çekirdek modül sürümü` sabit `1.0.0`'dır, bayat paketi göstermez.
- [ ] `Sızıntı koruyucuları: TAMAM`.
- [ ] B1 = 53,66 / 74,03 / 38,37 / 48,53 / 26,59 / 47,46 / 64,68 / 9,49 (test, `reports/local_bm25_test.json`; yalnız
      sağlama, BM25 deterministiktir). Değilse sonuçların hiçbiri yorumlanmaz.
- [ ] Sorgu sayıları: §10.1 val tablosunda her satırda `sorgu = 438`, §11 ablation'da her satırda `sorgu = 453`.
- [ ] Aşama 1 sıfır-atışı geçiyor: `val_sinyalleri` → `bi_val_en_iyi` × 100 > `val_zero_shot_base(WARM)` → `NDCG@10`
      (aynı hesap: val, WARM; log 0–1, sözlük yüzde) ve `VAL_COMP` → "İnce ayar (val)" ΔNDCG@10 > 0. Testteki
      "Ref: Zero-shot base bi-encoder" ve "İnce ayar (dense: …)" satırları yalnız kayıttır.
- [ ] §9.1 `aday_tavani_val(WARM)` NDCG@10 not edildi ve eşiğe göre okundu:
  - < 92: Aşama 2 işine başlama; önce Aşama 1 / `N_CANDIDATES` / RRF;
  - 92–95: gri bölge; Aşama 1 / aday iyileştirmesi muhtemelen hâlâ en verimlisi;
  - ≥ 95: Aşama 2 işi karşılık verebilir.

  Kıyas: val WARM BM25 ilk-50 84,14 / ilk-100 90,67 (`reports/metric_ceilings.json` → `aday_kumesi_tavani(BM25)` →
  `val|WARM|bm25_ilk{K}_kahin_yeniden_siralama`); hücre bunları `Kıyas (val, WARM, kâhin NDCG@10): …` satırında
  yazdırır. Val tavanları test tavanlarından birkaç puan düşüktür; `aday_tavani_test(tam_havuz,yalniz_rapor)` yalnız rapor içindir.
- [ ] Parite ≈ %100 (§13+).

## D. Çalıştırmadan sonra analiz (bu sırayla)

1. [ ] Dosyalar `runs/<tarih>_<etiket>/` altına kondu (`02` §7). `MyDrive/cv_ilan_sota/` her çalıştırmada üzerine
   yazılır (`figures/` birleştirilir, eski PNG'ler kalır); bu yüzden önce `run_summary.json` → `paket_parmak_izi`,
   `cfg`, `toplam_sure_dk` çalıştırılmış `.ipynb`'deki §2, CFG ve §13 çıktılarıyla karşılaştırıldı. Tutmuyorsa
   dosyalar bu çalıştırmanın değildir.
2. [ ] **Val sinyalleri önce:** `run_summary.json` → `val_sinyalleri` (`bi_val_en_iyi`, `ce_val_en_iyi`,
   `ltr_cv_ndcg10(val,WARM)`, τ taraması), `val_zero_shot_base(WARM)`, `val_tam_hat(WARM)`, `val_bilesen_testleri`,
   `tani` → `aday_tavani_val(WARM)`. Eğriler `stage1_training_log.csv` / `stage2a_training_log.csv`'dedir
   (`run_summary` yalnız en iyi değeri tutar); `scripts/13` ölçümleri `reports/lexical_lever_probe_val.json`'dadır.
   İki çalıştırma `per_query_val.csv` üzerinden eşleştirilmiş randomizasyonla kıyaslanır
   (`core.paired_randomization_test`). Hipotez doğrulandı mı?
3. [ ] **Tavanlar:** Val sonuçları val tavanlarıyla yan yana yazıldı: §9.1 `aday_tavani_val(WARM)`; WARM BM25
   ilk-50 / ilk-100 → kâhin 84,14 / 90,67; WARM kâhin P@10 83,22 (`metric_ceilings.json` → `kahin` →
   `val|warm|esik2`). Test sonuçları (kayıt) test tavanlarıyla: kâhin P@10 83,38, ikili kusursuz NDCG@10 91,36 (ikisi
   de tam havuz), `aday_tavani_test(tam_havuz,yalniz_rapor)`. Val ve test tavanları birbiriyle kıyaslanmaz.
4. [ ] **Bileşen testleri (val):** `val_component_tests.csv` (§10.1 `VAL_COMP`) okundu: ince ayar, BM25 füzyonu, CE,
   LambdaMART (kat dışı) katkı sağlıyor mu; ΔNDCG@10, ΔTop1 ve p-değerleri ne? `ablation_component_tests.csv` testtir,
   yalnız kayıt.
5. [ ] **Hata kovaları (val):** `per_query_val.csv` (0–1; `n_rel` WARM alakalı sayısıdır; `sistem` sütununda altı
   sistem var: "B1: BM25", "Ref: Zero-shot base bi-encoder", "Ours: Fine-tuned dense (tek)", "S1: Fine-tuned
   bi-encoder + BM25 (RRF)", "Ours: Hybrid + Cross-encoder", "S2: Hybrid + Cross-encoder + LambdaMART (kat dışı)") +
   CV altın sütunları (`clean_cvs.parquet`; kovalamak için değerlendirme tarafında altın kullanmak serbest). GPU'suz
   alternatif: `scripts/13` tarzı kat dışı tahminler (`04` §1; `scripts/13` yalnız özet yazar, kovalar için önce ona
   sorgu bazlı çıktı ekle). `per_query_test.csv` yalnız kayıt ve final raporun hata analizi içindir (`04` §5);
   kaldıraç ya da hedefli veri ondan seçilmez. Kovalar:
   - yaka, sektör, `n_rel` (<10 / 10–30 / >30);
   - grade-3'ü olan / olmayan sorgular;
   - `occ_key`'i train'de görülmüş / görülmemiş sorgular;
   - Remote / şehir.
6. [ ] **Hatalarda etiket gürültüsü:** Top-1 hatalarının bir örneğinde ilan ile CV elle okundu. Kural mı yanlış,
   model mi? Örnek val'den seçilir: `per_query_val.csv` → `sistem` = "S2: Hybrid + Cross-encoder + LambdaMART (kat
   dışı)", `Top1 == 0` (dosyada ilan kimliği yoktur). Top-1 ilanı bulmak için paket CLI'ı (Drive'daki
   `export_model_bundle.zip` açılmış olmalı): `python match_candidate_sota.py --bundle <paket> --cv-id <qid> --json`;
   çıktı yalnız sırayı, `job_key`'i, başlık/şirket/sektör/kademe/şehri ve skorları verir, ilan metnini vermez. İlan
   metnini `job_key` ile `data_bundle/clean_jobs.parquet` → `description` / `job_struct`'tan, CV metnini
   `clean_cvs.parquet` → `cv_text`'ten oku (paketteki LambdaMART tüm val'le eğitildi ve havuzda cold ilanlar da var;
   sıra kat dışı sıradan farklı olabilir). §12'deki test örnekleri yalnız rapor içindir. Kural şüphesi kullanıcıya
   sorulur (`00` §4).
7. [ ] **Test sayısı:** `test_degerlendirme_sayisi` (şu anki notebook'ta 14) ve şimdiye kadarki tam çalıştırma sayısı
   günlüğe yazıldı.
8. [ ] Sonuç ve bir sonraki adım günlüğe yazıldı; bir sonraki adım val sinyallerine dayanıyor. Hipotez tutmadıysa
   bu da bir sonuçtur; neden tutmadığı yazıldı.

## E. Bir sonuç iddia etmeden önce (rapor, savunma, kullanıcıya özet)

- [ ] Her sayı bir dosyaya dayanıyor (yol + anahtar). Ekrandan kopyalanan sayı yok.
- [ ] NDCG@10 için %95 bootstrap GA var. Karşılaştırmalarda eşleştirilmiş randomizasyon p-değeri var (val:
      `val_component_tests.csv`; iki çalıştırma arasında `per_query_val.csv`).
- [ ] Metrik tanımı (`REL_THRESHOLD=2`, IDCG havuzdaki tüm derecelerden, P@10 = isabet/10, alakalısı olmayan sorgular
      atlanır) değişmedi. Test tam havuzda, val karar sinyalleri WARM havuzda ölçülür. Değiştiyse kullanıcı onayı ve
      gerekçesi yazıldı; eski ve yeni sayılar ayrı raporlandı.
- [ ] Hedefler tavanlarıyla birlikte verildi. P@10 > 85'in testte yapısal olarak ulaşılamaz olduğu (kâhin 83,38)
      açıkça söylendi.
- [ ] Test kümesinin kaç kez okunduğu (çalıştırma sayısı) yazıldı, karar sürecinin val'e dayandığı belirtildi.
      Hazırlık aşamasındaki test temasları için `06` E000'a bak. Val sayıları raporlanıyorsa hafifçe iyimser oldukları
      (§10.1) söylendi.
- [ ] Etiketlerin kural türevi olduğu ve üst derecelerde cömert olduğu (kör doğrulama,
      `reports/qrels_validation.json`) söylendi.
- [ ] Sızıntı kapısı geçti. Hiçbir öznitelik altın etiketten türetilmedi.

## F. Sızıntı değişmezleri (asla bozulmaz)

`metadata.json` = `data_bundle/metadata.json`; aynı içerik repoda `reports/faz2_veri_ozeti.json` olarak da durur
(taze klonda `data_bundle/` yoktur).

| Değişmez | Nerede uygulanıyor | Nasıl kontrol edilir |
|---|---|---|
| Train/val/test CV kimlikleri ayrık | `05` kapısı (`id_kesisim_*`), notebook §2 assert | `metadata.json` → `sizinti_kontrolleri` |
| Normalize unvanlar ve neredeyse-kopya grupları bölümler arası ayrık | `05` kapısı (`unvan_kesisim_*`, `test_cv_cos_ge_0.8_trainde_ikizi`) | aynı |
| Sentetik CV yalnız train'de; val/test metninden üretilmez; test'e cos ≥ 0,8 ikizi yok. Yeni sentetik veri ya da LLM ile yeniden anotasyon kullanıcı kararıdır; üretim yalnız ilan tarafındaki WARM bilgiyi kullanır, hedef uzmanlıklar teste bakılarak seçilmez | `05` kapısı (`sentetik_*`), notebook assert, `06 merge` | aynı; val'e benzerlik kapıda yok (aşağıda) |
| Cold ilanlar (301) hiçbir eğitim veya seçim sinyalinde yok | `05` kapısı (`triplette_cold_ilan`, `legacy_pairs_cold_ilan`); notebook: madencilik, CoSENT, CE listeleri, LTR satırları, değerlendiriciler (`WARM_COLS`), §9.1 `aday_tavani_val(WARM)` ve §10.1 (cold ne adayda ne qrels'te); `scripts/13` (WARM aday, WARM qrels) | Kod incelemesi; yeni eğitim verisi ya da yeni val sinyali eklerken `WARM` filtresi, adaylarda cold maskesi ve `cols=WARM_COLS` |
| `gold_*` ve `legacy_*` sütunları (ilan tarafı dahil) öznitelik değildir; kullanıcıya sorulacak bir seçenek de değildir (`00` §4 Asla) | `core.assert_no_gold_columns` (ad denetimi), `metadata.json` → `sutun_rolleri` | Yeni öznitelik ekleyince adına **ve kaynağına** bak (denetim tarafsız adı yakalamaz) |
| Train CV altın etiketleri yalnız train'de sınıflandırıcı **hedefi** olarak kullanılır | `AttributeModels.fit` (train CV'leri) | Yeni sınıflandırıcı da yalnız train ile eğitilmeli |
| Vitrin CV'leri (20951, 10587, 10511, 21029) testte | `05` (`SHOWCASE` grupları teste zorlanır) | `metadata.json` → `vitrin_kontrol.<cv_id>.split` == `"test"` (dört CV; şu an 4/4). `vitrin_adaylari_test` yalnız `SHOWCASE` sabitinin kopyasıdır, bölümü kanıtlamaz. Notebook §2 testte olmayan kimliği uyarmadan atar (`Vitrin (test) adayları:` satırında dört aday olmalı) |
| qrels doğrulama yargıları eğitime girmez | Protokol | `annotation/qrels_validation/` hiçbir eğitim kodunda okunmaz |
| Test'te seçim yapılmaz | Protokol (kodda tüm seçimler val, WARM) | Günlükte her kararın §A listesinden bir val gerekçesi var |

Kapı listesi `scripts/05_build_dataset.py` → `leakage_checks()` ve `main()` içindedir (14 sayım kontrolü, hepsi 0).
Kapıda olmayan ama şu an ölçülerek 0 olanlar (hiçbiri dosyaya kayıtlı değildir; sayımlar ve benzerlik değerleri
yerelde `leakage_checks()`'teki yöntemle ölçüldü, bunlara dayanmadan önce yeniden ölç):
- val ↔ test unvan kesişimi;
- val triplet'lerinde cold ilan (`triplette_cold_ilan` yalnız train triplet'lerine bakar);
- val → train neredeyse-kopya (gerçek CV'ler): gruplama gereği 0 (unvan maskeli TF-IDF ≥ 0,8 aynı gruba girer);
- sentetik train → val neredeyse-kopya: gruplama sentetik CV'leri kapsamaz, kapı yalnız sentetik → test'e bakar.
  Hazırlıkta yerelde ölçülüp 0 bulundu ama dosyaya yazılmadı; gerekiyorsa `leakage_checks()` yöntemiyle ölç ve
  bir dosyaya yaz (sentetik veri yeniden üretilirse mutlaka).

Ayrıca kapıda yok: vitrin CV'lerinin testte olduğuna dair ayrı bir kapı (`vitrin_kontrol` yalnız kayıt tutar).

Bu boşlukları kapıya eklemek senin kararın (kapıyı sıkılaştırmak serbesttir, gevşetmek değil). Eklemek
`metadata.json`'u, dolayısıyla §2 parmak izini değiştirir: kullanıcı yeni `data_bundle.zip`'i yükler.
