# Meslek Anotasyon Kılavuzu (ISCO-08 + Uzmanlık + Yaka + Kıdem)

Bu kılavuz, 3.043 CV ve 3.000 iş ilanının **aynı ölçütlerle** etiketlenmesi için yazılmıştır.
Etiketler, sistemin değerlendirmesinde kullanılan **kademeli alaka (graded relevance, 0–3)**
ground-truth'unun meslek bileşenini oluşturur. Etiketler modele **özellik olarak verilmez**;
yalnızca qrels (doğru cevap anahtarı) üretiminde kullanılır.

Her öğe (CV veya İLAN) için aşağıdaki alanlar üretilir:

| Alan | Tip | Açıklama |
|---|---|---|
| `id` | str | Girdi kimliği (C10000, J0001 ...) — aynen kopyalanır |
| `isco4` | str | ISCO-08 dört haneli birim grup kodu, örn. `"7512"` |
| `isco_title` | str | ISCO-08 birim grubunun İngilizce resmi adı, örn. `"Bakers, pastry-cooks and confectionery makers"` |
| `specialty` | str | Kanonik Türkçe uzmanlık adı (aşağıdaki kurallar) |
| `collar` | `"BLUE"` / `"WHITE"` | Mavi / beyaz yaka |
| `seniority` | int 0–5 | Kıdem ölçeği (aşağıda) |
| `conf` | int 1–3 | Güven: 3 = kesin, 2 = makul, 1 = belirsiz |

## 1. ISCO-08 kodu (`isco4`)

* Kişinin/ilanın **fiilen yaptığı işi** kodlayın; unvan yanıltıcı olabilir (örn. "Kontrolör" maliyet
  kontrolörü ise 2411 Accountants, kalite kontrolör ise 3119/7543 vb.). Bağlam (özet, beceriler,
  deneyim, ilan açıklaması) belirleyicidir.
* ISCO-08 resmi 4 haneli birim gruplarını kullanın (ILO, 2012). Kod uydurmayın.
* Sektör (`industry`) yalnızca bağlamdır; kodu sektör değil meslek belirler. Bir inşaat
  firmasındaki muhasebeci 2411'dir.
* Yönetim: işin ağırlığı **yönetmek** olan müdür/direktör/genel müdür rolleri Major Group 1
  (11xx–14xx) altında kodlanır. İşi bizzat yapan takım lideri / şef / ustabaşı kendi
  mesleğinin kodunu alır (ISCO kuralı: ilk kademe denetçiler çoğunlukla denetledikleri işin
  koduyla ya da 312x denetçi kodlarıyla sınıflanır — inşaat şantiye şefi 3123, üretim vardiya
  amiri 3122).
* Tutarlılık için **ev kuralları** (sık belirsizlikler):
  * Yazılım geliştirici (backend, frontend, mobil, full-stack, oyun, gömülü yazılım) → **2512**;
    Web geliştirici de 2512. Kodlamayan web/UI/UX tasarımcı → **2166**.
  * Yazılım test/QA → **2519**. DevOps / SRE / bulut / sistem yöneticisi → **2522**.
    Ağ mühendisi → **2523**. Veritabanı yöneticisi (DBA) → **2521**. Siber güvenlik → **2529**.
  * Veri bilimci / makine öğrenmesi / yapay zekâ mühendisi → **2120**; iş/veri analisti
    (raporlama, BI) → **2511**; veri mühendisi (ETL, pipeline) → **2521**.
  * BT/yazılım yöneticileri (CTO, BT müdürü, yazılım geliştirme müdürü) → **1330**.
  * Muhasebe uzmanı / mali müşavir / maliyet kontrolörü → **2411**; ön muhasebe / muhasebe
    elemanı (büro işi) → **4311**; finansal analist / yatırım → **2413**; denetçi (audit) → **2411**.
  * İnsan kaynakları uzmanı / işe alım / bordro uzmanı (uzman düzey) → **2423**; bordro/özlük
    memuru (büro) → **4416**.
  * Satış temsilcisi / bölge satış / B2B satış / distribütör ilişkileri → **3322**;
    teknik satış → **2433**; mağaza satış danışmanı → **5223**; kasiyer → **5230**.
  * Müşteri hizmetleri / çağrı merkezi temsilcisi → **4222**.
  * Hava trafik kontrolörü (ATC) → **3154**. Uçuş harekât uzmanı / uçuş dispeçeri
    (flight dispatcher, flight operations) → **4323** (Transport clerks). Distribütör / bayi
    satış yöneticisi → **3322** ya da yönetim ağırlıklıysa **1221**.
  * Hemşire → **2221**; ebe → **2222**; hasta bakıcı → **5321**; laboratuvar teknikeri → **3212**.
  * Aşçı → **5120**; mutfak şefi / aşçıbaşı → **3434**; pastacı / baklavacı / fırıncı → **7512**;
    kasap → **7511**; garson → **5131**; barista/barmen → **5132**; bulaşıkçı / komi → **9412**.
  * Kaynakçı (TIG, MIG/MAG, elektrik ark) → **7212**; CNC/torna/freze operatörü ve
    tezgâhçı → **7223**; forklift operatörü → **8344**; vinç operatörü → **8343**;
    iş makinesi operatörü (ekskavatör, loder, greyder) → **8342**.
  * Sıvacı → **7123**; duvarcı → **7112**; boyacı (bina) → **7131**; elektrikçi (bina) → **7411**;
    tesisatçı → **7126**; döşemeci (mobilya) → **7534**.
  * Şoför: tır/kamyon → **8332**; servis/otobüs → **8331**; binek/makam şoförü → **8322**;
    kurye (motorlu) → **8321**; yaya/bisikletli kurye → **9621**.
  * Güvenlik görevlisi → **5414**; temizlik görevlisi → **9112**; depo / yükleme elemanı → **9333**;
    depo sorumlusu (stok kâtibi) → **4321**.
  * Öğretmenler: okul öncesi → **2342**; ilkokul → **2341**; ortaöğretim branş → **2330**;
    dil öğretmeni (kurs) → **2353**; üniversite → **2310**.
  * Avukat → **2611**; hukuk müşaviri (şirket içi) → **2611**.
* Yukarıda olmayan durumlarda en doğru ISCO-08 birim grubunu seçin.

## 2. Uzmanlık (`specialty`)

Bir İK uzmanının "aynı iş" diyeceği düzeyde, **kanonik Türkçe meslek adı**:

* Kıdem/rütbe sözcüklerini atın: Stajyer, Junior, Kıdemli, Senior, Baş-, Yardımcı(sı),
  Uzman Yardımcısı, Lead, Şef (yalnız rütbe anlamındaysa) → bunlar `seniority` alanına gider.
  "Aşçı Yardımcısı" → specialty **"Aşçı"**, seniority 1. "Kıdemli Java Geliştirici" →
  **"Backend Yazılım Geliştirici"**, seniority 3.
* Firma, şehir, vardiya, sözleşme tipi, parantez içi İngilizce tekrarları atın.
* **Teknoloji/marka yığını ayrımı yapmayın** (Java vs Python backend → aynı
  "Backend Yazılım Geliştirici"; React vs Vue → "Frontend Yazılım Geliştirici").
* **Zanaat/ehliyet/ekipman ayrımı yapın**: "TIG (Argon) Kaynakçısı" ≠ "Gazaltı (MIG/MAG)
  Kaynakçısı"; "CNC Torna Operatörü" ≠ "CNC Freze Operatörü"; "Kule Vinç Operatörü" ≠
  "Mobil Vinç Operatörü"; "Baklava Ustası" ≠ "Pide/Lahmacun Ustası"; "Sıva Ustası" ≠
  "Seramik/Fayans Ustası"; "Uçuş Harekat Uzmanı (Dispeçer)" ≠ "Distribütör Satış Yöneticisi".
* Yöneticiler: "X Müdürü" / "X Direktörü" → specialty "X Müdürü" (müdür ve direktör aynı
  specialty, fark `seniority` ile 4 vs 5).
* Yazım: Türkçe büyük harfle başlayan sözcükler, tekil, kısaltmasız ("Mühendis", "Mühendisi").
* Aynı parçadaki benzer öğelere **aynı specialty metnini birebir** verin.

## 3. Yaka (`collar`)

* **BLUE**: bedensel/saha/üretim/zanaat işi; makine/araç/iş makinesi kullanımı; mutfak ve
  servis personeli (aşçı, mutfak şefi, garson, barista); temizlik; güvenlik görevlisi; bakım
  ve hasta bakım personeli; depo/lojistik elemanı; kasiyer; resepsiyonist; kabin memuru;
  çağrı merkezi temsilcisi; **teknisyen ve tekniker** (bu veri setinin konvansiyonu).
* **WHITE**: ofis/bilgi/uzman/yönetici işi; mühendis, yazılımcı, analist, muhasebe, İK, hukuk,
  pazarlama, satış temsilcisi/yöneticisi, müşteri temsilcisi (saha dışı, kurumsal), öğretmen,
  doktor, hemşire, eczacı, psikolog, pilot, mimar, tasarımcı, ön muhasebe/idari büro işi,
  şantiye şefi (mühendis), depo sorumlusu/şefi.

## 4. Kıdem (`seniority`, 0–5)

| Değer | Anlam |
|---|---|
| 0 | Stajyer, öğrenci, çırak (henüz mezun olmamış / staj arayan) |
| 1 | Giriş seviyesi: yeni mezun, junior, yardımcı eleman, 0–2 yıl |
| 2 | Orta seviye uygulayıcı / uzman / kalfa, ~2–5 yıl (**varsayılan**) |
| 3 | Kıdemli: senior, usta (deneyimli zanaatkâr), başuzman, ~5+ yıl |
| 4 | Lider: takım lideri, şef, ustabaşı, sorumlu, birim/departman müdürü |
| 5 | Üst yönetim: direktör, genel müdür, C-level, ülke/bölge direktörü |

* **CV**: açık yıl bilgisini (deneyim tarihleri, "X yıllık deneyim") unvan sözcüklerinden
  önce kullanın. Tarihler maskeliyse ([TARİH]) unvan + özet ifadelerine bakın ("tecrübeli",
  "çekirdekten yetişme usta", "son sınıf öğrencisi"). Bilgi yoksa 2.
* **İLAN**: `position_level` alanından başlayın (INTERN→0; ENTRY/ENTRY_LEVEL/NEW_GRADUATE/
  ASSISTANT→1; MID/MID_LEVEL/SPECIALIST/ASSISTANT_SPECIALIST→2; SENIOR/SENIOR_SPECIALIST→3;
  LEAD/TEAM_LEAD/MANAGER→4; DIRECTOR/EXECUTIVE→5; UNSPECIFIED→açıklamadan çıkarın) ve
  yalnızca unvan/gereksinimler açıkça çelişiyorsa düzeltin (ör. "Direktör" unvanlı LEAD → 5;
  "en az 8 yıl" isteyen SPECIALIST → 3).

## 5. Çıktı biçimi

Tek bir JSON listesi; girdi parçasındaki **her öğe için tam bir kayıt**, aynı `id` ile:

```json
[{"id": "C20951", "isco4": "7512", "isco_title": "Bakers, pastry-cooks and confectionery makers",
  "specialty": "Baklava Ustası", "collar": "BLUE", "seniority": 3, "conf": 3}]
```
