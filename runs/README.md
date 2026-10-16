# runs/

Her Colab A100 çalıştırmasının **küçük** çıktıları buraya konur. Yerel (GPU'suz) val ölçümleri de aynı adlandırmayla
buraya girer (ör. `runs/<tarih>_bm25-top1-hatalari-val/`: JSON + `NOTLAR.md`). Model ağırlıkları ve
`export_model_bundle.zip` (≈2,3 GB, tahmin; gerçek boyutu §13'ün `Paket: … GB` satırı yazar) repoya girmez. Drive'da
(`MyDrive/cv_ilan_sota/`) yalnız §13'e ulaşan ve Drive'ı bağlı olan **son** çalıştırmanın paketi durur: klasör
sabittir, §13 her çalıştırmada aynı adlarla üzerine yazar ve klasörü hiç temizlemez. Bir çalıştırmanın modeli
saklanacaksa kullanıcı zip'i Drive'da yeniden adlandırır (ör. `export_model_bundle_<etiket>.zip`).

## Yapı

```
runs/
  2026-10-20_ilk-calistirma/          # <YYYY-MM-DD>_<kısa-etiket>
    NOTLAR.md                          # 3-10 satır: amaç, CFG farkı, val sonucu, sonraki adım (06_DENEY_GUNLUGU ile aynı kimlik)
    executed.ipynb                     # Colab: Dosya → İndir → .ipynb (çıktılarıyla; aşağıdaki eşleşme kontrolü için şart)
    run_summary.json                   # karar: val_sinyalleri, val_zero_shot_base(WARM), val_tam_hat(WARM),
                                       #   val_bilesen_testleri, tani → aday_tavani_val(WARM); yalnız kayıt: hedefler,
                                       #   ana_ablation; ayrıca hucre_sureleri, sureler_sn, toplam_sure_dk,
                                       #   gpu_tepe_bellek_GB, paket_parmak_izi, surumler, cfg
    stage1_training_log.csv            # karar: Aşama 1 val eğrisi (eval_val_*)
    stage2a_training_log.csv           # karar: Aşama 2a (CE) val eğrisi
    val_pipeline.csv                   # karar: §10.1 VAL_TBL (WARM havuz, LambdaMART kat dışı, NDCG %95 GA)
    val_component_tests.csv            # karar: §10.1 VAL_COMP (ΔNDCG@10, ΔTop1, p)
    per_query_val.csv                  # karar: sorgu bazlı val (0–1, `sistem` sütunu): hata kovaları, iki çalıştırma arası eşleştirme
    per_query_test.csv                 # yalnız kayıt: final raporun hata analizi
    ablation_test.csv                  # yalnız kayıt (§11)
    ablation_component_tests.csv       # yalnız kayıt (§11)
    cold_start_test.csv                # yalnız kayıt (§11.1)
    figures/                           # görseller (eski dosya tuzağı: aşağıda)
```

§10.1 dosyaları hafifçe iyimserdir (val erken durdurma, checkpoint ve LTR hiperparametre seçiminde kullanıldı):
çalıştırmaları birbiriyle kıyaslamak içindir, yansız kestirim değildir.

## Bu dosyalar bu çalıştırmanın mı?

- Drive'a yazma yalnız §13'tedir; CSV'ler ve `run_summary.json` yalnız varsa kopyalanır. §13'e ulaşmayan bir
  çalıştırmada `MyDrive/cv_ilan_sota/` içindeki her şey önceki çalıştırmaya aittir. Drive bağlanmadıysa da (§2:
  `Drive bağlanmadı (sorun değil, yerel diskle devam)`) §13 Drive'a hiçbir şey yazmaz; bu çalıştırmanın dosyaları
  yalnız Colab'daki `/content/outputs/` (ve `/content/export_model_bundle.zip`) içindedir, oradan indirilir.
- `figures/` Drive'a birleştirilerek kopyalanır (`copytree(..., dirs_exist_ok=True)`), eski PNG'ler kalır: ör.
  `TUNE_TEMPERATURE=False` iken eski `temperature_sweep.png`, vitrin CV'leri değişince eski `attention_<cv_id>.png` /
  `shap_<cv_id>.png`. Aynı Colab makinesinde `/content/outputs` da temizlenmez; eski görseller oradan pakete
  (`export_model_bundle/figures/`) ve Drive'a karışır. Önceki çalıştırmanın dosyaları `runs/`'a alındıktan **sonra**
  ve yeni çalıştırmadan önce kullanıcıdan `MyDrive/cv_ilan_sota/figures/` klasörünü silmesini, aynı makinedeyse
  `!rm -rf /content/outputs` çalıştırmasını iste.
- Dosyalar bir sonraki çalıştırmadan **önce** indirilir ya da kopyalanır; sonraki çalıştırma onları ezer.
- `runs/`'a koymadan önce eşleştir: `run_summary.json` → `paket_parmak_izi` = `.ipynb` §2'deki `paket parmak izi
  (sha256[:12])` (çalıştırmaya verilen paketin `sha256sum data_bundle/metadata.json match_candidate_sota.py | cut -c1-12`
  değeriyle de aynı); `cfg` = CFG hücresinin çıktısı; `toplam_sure_dk` = §13'ün `toplam süre: … dk` satırı. Biri
  tutmuyorsa dosyalar bu çalıştırmanın değildir.

## Kurallar

- Etiket kısa ve anlamlı olmalı: `ilk-calistirma`, `k100`, `isco-oznitelik`, `cosent-yok`.
- `NOTLAR.md` ilk satırda deney kimliğini verir (`E001` gibi; bkz. `ajan/06_DENEY_GUNLUGU.md`).
- Karar yalnız val sinyallerine dayanır (WARM havuz): yukarıda "karar" işaretli dosyalar, buraya aynı adlandırmayla
  konan yerel val ölçümleri (ör. `bm25_top1_hatalari_val.json`) ve `reports/lexical_lever_probe_val.json`
  (`scripts/13`). Test dosyaları yalnız kayıttır; bir sonraki kaldıraç ya da hedefli veri onlara bakılarak seçilmez.
  Devir hazırlığında testte yapılan keşif ölçümleri (`06` E000) buraya kanıt olarak konmaz.
- Bir çalıştırmanın çıktıları sonradan düzenlenmez. Yanlışsa yeni bir not ekle.
- Kullanıcıdan gelen dosyaları okumadan commit etme. Kişisel veri ya da gizli anahtar içeren bir şey varsa çıkar.
  CV metinleri zaten veri setinde var, ama dışarıdan eklenen yeni CV'ler kullanıcıya sorulmadan repoya girmez.
