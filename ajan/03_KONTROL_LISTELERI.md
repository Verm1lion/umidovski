# Kontrol listeleri

Kısa tutuldu, çünkü her seferinde okunacaklar. İşaretlemeden geçme.

## A. Bir değişikliğe başlamadan önce

- [ ] Hipotez `06_DENEY_GUNLUGU.md`'ye yazıldı: ne değişiyor, hangi val sinyali hangi yönde ne kadar oynamalı,
      ve neden (mekanizma).
- [ ] Değişiklik **tek** bir şey. Birden çok şey değişiyorsa ayrı deneylere bölündü.
- [ ] Karar hakkı kontrolü yapıldı (`00` §4). Metrik tanımı, alaka kuralı, ilan altın etiketi, dış API veya lisans
      işin içindeyse önce kullanıcıya soruldu.
- [ ] Kararın dayandığı sinyal val'den geliyor: Aşama 1/2a log'ları, `ltr_cv_ndcg10`, §9.1 val tavanı, `scripts/13`
      tarzı val GroupKFold. Testten gelmiyor.

## B. Çalıştırmadan önce (kod değiştiyse)

- [ ] Notebook değişikliği `scripts/07_make_notebook.py` içinde yapıldı. `.ipynb` dosyası
      `python scripts/07_make_notebook.py` ile üretildi.
- [ ] `match_candidate_sota.py` değiştiyse `python scripts/05_build_dataset.py` çalıştırıldı. Sızıntı kapısı geçti
      ve `data_bundle.zip` yenilendi. Zip'teki çekirdek repo çekirdeğiyle aynı:
      `unzip -p data_bundle.zip data_bundle/match_candidate_sota.py | cmp - match_candidate_sota.py`.
- [ ] Veri üreticisi değişmediyse parquet'ler birebir aynı. Değiştiyse bu bilinçli bir karar ve günlüğe yazıldı.
- [ ] `python scripts/08_dry_run_notebook.py --bundle data_bundle.zip --workdir work/dry_run --kernel <k>` →
      `reports/notebook_dogrulama.json` içinde `"durum": "OK"`, `RUN_OK: true`, `export_zip: true`. pyflakes listesi boş.
- [ ] Yeni öznitelik eklendiyse:
  - çekirdekte `FEATURE_COLUMNS` güncellendi;
  - öznitelik `build_pair_features` içinde hesaplanıyor;
  - `SOTAMatcher` aynı özniteliği çıkarımda da üretiyor;
  - yasaklı ad kullanılmadı (`gold_*`, `legacy_*`, `grade`, `label`, …).
- [ ] Kullanıcıya şunlar söylendi: hangi dosyalar yüklenecek, `/content/data_bundle` silinmeli mi, çalıştırmanın
      amacı (tek cümle).

## C. Çalışma sırasında / hemen sonra

`02` §3'teki sağlık tablosunu satır satır geç. Özellikle şunlar:
- [ ] B1 = 53,66 / 74,03 / 38,37 / 48,53 / 26,59 / 47,46 / 64,68 / 9,49. Değilse sonuçların hiçbiri yorumlanmaz.
- [ ] `Sızıntı koruyucuları: TAMAM`.
- [ ] Ablation'da her satırda `sorgu = 453`.
- [ ] İnce ayarlı dense, val'de sıfır-atış BGE-M3'ü geçiyor.
- [ ] §9.1 val aday tavanı not edildi.
- [ ] Parite ≈ %100.

## D. Çalıştırmadan sonra analiz (bu sırayla)

1. [ ] Dosyalar `runs/<tarih>_<etiket>/` altına kondu (`02` §7).
2. [ ] **Val sinyalleri önce:** `run_summary.json` → `val_sinyalleri`, `tani`. Hipotez doğrulandı mı?
3. [ ] **Tavanlar:** Sonuç kâhin (83,38 P@10), ikili kusursuz (91,36 NDCG@10) ve §9.1 aday tavanı ile yan yana yazıldı.
4. [ ] **Bileşen testleri:** `ablation_component_tests.csv` okundu. Her bileşen (ince ayar, BM25 füzyonu, CE,
      LambdaMART) katkı sağlıyor mu ve p-değerleri ne?
5. [ ] **Hata kovaları** (`per_query_test.csv` + CV altın sütunları; değerlendirme tarafında altın kullanmak serbest):
   - yaka, sektör, `n_rel` (<10 / 10–30 / >30);
   - grade-3'ü olan / olmayan sorgular;
   - `occ_key`'i train'de görülmüş / görülmemiş sorgular;
   - Remote / şehir.
6. [ ] **Hatalarda etiket gürültüsü:** Top-1 hatalarının bir örneğinde ilan ile CV elle okundu. Kural mı yanlış,
      model mi?
7. [ ] **Test sayısı:** `test_degerlendirme_sayisi` ve şimdiye kadarki tam çalıştırma sayısı günlüğe yazıldı.
8. [ ] Sonuç ve bir sonraki adım günlüğe yazıldı. Hipotez tutmadıysa bu da bir sonuçtur; neden tutmadığı yazıldı.

## E. Bir sonuç iddia etmeden önce (rapor, savunma, kullanıcıya özet)

- [ ] Her sayı bir dosyaya dayanıyor (yol + anahtar). Ekrandan kopyalanan sayı yok.
- [ ] NDCG@10 için %95 bootstrap GA var. Karşılaştırmalarda eşleştirilmiş randomizasyon p-değeri var.
- [ ] Metrik tanımı (`REL_THRESHOLD=2`, tam havuz, P@10 = isabet/10, alakasız sorgular atlanır) değişmedi.
      Değiştiyse kullanıcı onayı ve gerekçesi yazıldı; eski ve yeni sayılar ayrı raporlandı.
- [ ] Hedefler tavanlarıyla birlikte verildi. P@10 > 85'in testte yapısal olarak ulaşılamaz olduğu (kâhin 83,38)
      açıkça söylendi.
- [ ] Test kümesinin kaç kez okunduğu (çalıştırma sayısı) yazıldı, karar sürecinin val'e dayandığı belirtildi.
      Hazırlık aşamasındaki test temasları için `06` E000'a bak.
- [ ] Etiketlerin kural türevi olduğu ve üst derecelerde cömert olduğu (kör doğrulama) söylendi.
- [ ] Sızıntı kapısı geçti. Hiçbir öznitelik altın etiketten türetilmedi.

## F. Sızıntı değişmezleri (asla bozulmaz)

| Değişmez | Nerede uygulanıyor | Nasıl kontrol edilir |
|---|---|---|
| Train/val/test CV kimlikleri ayrık | `05` kapısı (`id_kesisim_*`), notebook §2 assert | `metadata.json` → `sizinti_kontrolleri` |
| Normalize unvanlar ve neredeyse-kopya grupları bölümler arası ayrık | `05` kapısı (`unvan_kesisim_*`, `test_cv_cos_ge_0.8_trainde_ikizi`) | aynı |
| Sentetik CV yalnız train'de; val/test metninden üretilmez; test'e cos ≥ 0,8 ikizi yok | `05` kapısı (`sentetik_*`), notebook assert, `06 merge` | aynı |
| Cold ilanlar (301) hiçbir eğitim veya seçim sinyalinde yok | `05` kapısı (`triplette_cold_ilan`, `legacy_pairs_cold_ilan`); notebook: madencilik, CoSENT, CE listeleri, LTR satırları, değerlendiriciler (WARM_COLS) | Kod incelemesi; yeni eğitim verisi eklerken `WARM` filtresi |
| `gold_*` ve `legacy_*` sütunları öznitelik değildir | `core.assert_no_gold_columns` (ad denetimi), `metadata.json` → `sutun_rolleri` | Yeni öznitelik ekleyince adına **ve kaynağına** bak (denetim tarafsız adı yakalamaz) |
| Train CV altın etiketleri yalnız train'de sınıflandırıcı **hedefi** olarak kullanılır | `AttributeModels.fit` (train CV'leri) | Yeni sınıflandırıcı da yalnız train ile eğitilmeli |
| Vitrin CV'leri (20951, 10587, 10511, 21029) testte | `05` (`SHOWCASE` grupları teste zorlanır) | `metadata.json` → `vitrin_adaylari_test` |
| qrels doğrulama yargıları eğitime girmez | Protokol | `annotation/qrels_validation/` hiçbir eğitim kodunda okunmaz |
| Test'te seçim yapılmaz | Protokol (kodda tüm seçimler val) | Günlükte her kararın val gerekçesi var |

Kapı listesi `scripts/05_build_dataset.py` → `leakage_checks()` ve `main()` içindedir (14 sayım kontrolü, hepsi 0).
Kapıda olmayan ama şu an ölçülerek 0 olanlar:
- val ↔ test unvan kesişimi;
- val triplet'lerinde cold ilan;
- val → train neredeyse-kopya kontrolü (gruplama yapısı gereği 0);
- vitrin CV'lerinin testte olduğuna dair ayrı bir kapı.

Bu boşlukları kapıya eklemek senin kararın.
