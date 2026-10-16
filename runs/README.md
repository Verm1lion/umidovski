# runs/

Her Colab A100 çalıştırmasının **küçük** çıktıları buraya konur. Model ağırlıkları ve `export_model_bundle.zip`
(≈2,3 GB) repoya girmez; onlar Drive'da (`MyDrive/cv_ilan_sota/`) kalır.

## Yapı

```
runs/
  2026-10-20_ilk-calistirma/          # <YYYY-MM-DD>_<kısa-etiket>
    NOTLAR.md                          # 3-10 satır: amaç, CFG farkı, sonuç, sonraki adım (06_DENEY_GUNLUGU ile aynı kimlik)
    executed.ipynb                     # Colab: Dosya → İndir → .ipynb (çıktılarıyla)
    run_summary.json                   # hedef kararı, val sinyalleri, tanılar, süreler, bellek, sürümler, CFG
    per_query_test.csv                 # sistem × sorgu metrikleri
    stage1_training_log.csv
    stage2a_training_log.csv
    ablation_test.csv
    ablation_component_tests.csv
    cold_start_test.csv
    figures/
```

## Kurallar

- Etiket kısa ve anlamlı olmalı: `ilk-calistirma`, `k100`, `isco-oznitelik`, `cosent-yok`.
- `NOTLAR.md` ilk satırda deney kimliğini verir (`E001` gibi; bkz. `ajan/06_DENEY_GUNLUGU.md`).
- Bir çalıştırmanın çıktıları sonradan düzenlenmez. Yanlışsa yeni bir not ekle.
- Kullanıcıdan gelen dosyaları okumadan commit etme. Kişisel veri ya da gizli anahtar içeren bir şey varsa çıkar.
  CV metinleri zaten veri setinde var, ama dışarıdan eklenen yeni CV'ler kullanıcıya sorulmadan repoya girmez.
