# Deney günlüğü

Her deney bir kayıttır. **Hipotez çalıştırmadan önce yazılır**, sonuç sonra eklenir. Kayıtlar silinmez; yanlışsa
altına düzeltme notu düşülür. Test sayıları yalnız kayıt içindir; "Karar" satırı val sinyaline dayanmalıdır.

## Şablon

```markdown
### E0NN — <kısa ad>   (<YYYY-MM-DD>, runs/<klasör>)
- **Tek değişiklik:** <CFG anahtarı / kod yeri: eski → yeni>
- **Hipotez ve mekanizma:** <neden işe yaramalı>
- **Beklenen val etkisi:** <hangi sinyal, hangi yönde, ≈ne kadar> (ör. Aşama 1 eval_val_graded_ndcg@10 +2 puan)
- **Karar sinyali:** <eval_val_graded_ndcg@10 | CE val | ltr_cv_ndcg10 | §9.1 val aday tavanı | scripts/13 val>
- **Sonuç (val):** <sayı(lar), dosya yolu>
- **Test (yalnız kayıt):** <S2 NDCG@10 / Top-1 / P@10 [GA]>, test_degerlendirme_sayisi=<n>, toplam tam çalıştırma=<k>
- **Tavanlar:** aday tavanı val=<x>, ikili kusursuz=91,36 (test), kâhin P@10=83,38 (test)
- **Karar:** <tut / geri al / genişlet> — gerekçe (val'e dayalı)
- **Sonraki adım:** <tek cümle>
```

## Kullanıcı kararları kaydı

| Tarih | Soru | Karar | Etkisi |
|---|---|---|---|
| — | P@10 > 85 testte ulaşılamaz (kâhin 83,38). Hedef nasıl ifade edilsin? | *bekliyor* | — |

## Kayıtlar

### E000 — Hazırlık durumu ve test kümesiyle temaslar (2026-10-16)

- **Durum:**
  - Veri paketi, notebook ve çekirdek hazır; CPU kuru çalıştırma OK.
  - GPU'da hiç çalıştırma yok, nöral sistemlerin sonucu yok.
- **GPU'suz referanslar:**
  - BM25 test: NDCG@10 53,66 [51,5; 55,8], Top-1 64,68, P@10 38,37 (`reports/local_bm25_test.json`).
  - BM25 + LambdaMART (leksikal) test: 66,05 [63,9; 68,2], Top-1 72,85 (`reports/local_lexical_ltr.json`).
- **Tavanlar** (`reports/metric_ceilings.json`):
  - Kâhin P@10: test 83,38 / val 85,08.
  - İkili kusursuz NDCG@10: test 91,36 / val 91,96.
  - BM25 ilk-50 kâhin NDCG@10: test 86,95 / val 83,15.
- **Test kümesiyle temaslar (dürüst sayım):**
  1. Faz 2 raporu için BM25 ve leksikal LTR testte birer kez ölçüldü (yukarıdaki iki dosya).
  2. Tavanlar test etiketlerinden hesaplandı. Model yok, seçim yok; bunlar tanımın özellikleri.
  3. Devir hazırlığında bir analiz ajanı, leksikal hatta ≈14 LTR varyantını (ISCO-4 özniteliği, kNN, K=50/100,
     train sorgularıyla çapraz uydurma) ve BM25 Top-1 hata kategorilerini **test üzerinde** keşif amaçlı ölçtü.
     - Bu sayılar repoda değildir ve hiçbir karar için kullanılmamalıdır.
     - Yerine yalnız val kullanan `scripts/13_lexical_lever_probe.py` yazıldı (`reports/lexical_lever_probe_val.json`).
     - **Sonuç:** Leksikal varyantlar açısından test kümesi artık tamamen "bakir" değil. Nöral hat testte hiç ölçülmedi.
       Savunmada bu açıkça söylenmeli.
- **Karar:** İlk A100 çalıştırması varsayılan `CFG` ile yapılır (E001).

### E001 — İlk A100 çalıştırması, varsayılan CFG   (planlandı)

- **Tek değişiklik:** yok; referans çalıştırma.
- **Hipotez:**
  - Sağlık satırları `02` §3 ile uyumlu olacak; B1 birebir 53,66.
  - İnce ayarlı dense, val'de sıfır-atış BGE-M3'ü geçecek.
  - §9.1 val aday tavanı, BM25 ilk-50 tavanının (val 83,15) belirgin üstünde olacak.
- **Ölçülecekler:**
  - süre ve tepe bellek;
  - kırpılma oranları;
  - aday tavanı;
  - Aşama 1 ve CE val eğrileri;
  - bileşen testleri;
  - hedef satırları (kayıt için).
- **Karar sinyali:** §9.1 val aday tavanı. Bu tavan < 92 ise ilk iş Aşama 1 / `N_CANDIDATES` (`04` §1 düğüm 3).
