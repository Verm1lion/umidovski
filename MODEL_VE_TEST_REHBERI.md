# 🚀 CV - İş İlanı Eşleştirme ve Semantik Öneri Sistemi Rehberi

Bu doküman, projede geliştirilen **Semantik Eşleştirme Motoru**, kullanılan **derin öğrenme modeli**, **hibrit sıralama kuralları**, **test metrikleri** ve **dosya mimarisini** ekibe yeni katılan bir geliştiricinin kolayca anlayabilmesi için hazırlanmıştır.

---

## 📌 1. Projenin Amacı Nedir?

Geleneksel iş arama motorları sadece **anahtar kelime (keyword) araması** yapar. Örneğin bir adayın CV'sinde *"Python"* yazmıyorsa ama *"Django ve Flask ile backend API geliştirdi"* yazıyorsa, klasik filtreler adayı eler.

Bu projenin amacı:
* Adayın özgeçmişi (CV) ile şirketlerin iş ilanları arasındaki **anlamsal ilişkiyi (semantik benzerlik)** kavramak,
* Adayın sektörünü, yaka tipini (Mavi Yaka / Beyaz Yaka), kıdem seviyesini ve lokasyonunu dikkate alarak **en uygun ilk 10 iş ilanını (Top-10)** sıralamaktır.

---

## 🧠 2. Model Mimarisi: `multilingual-e5-small`

Sistemde metinleri matematiksel vektörlere (embedding) dönüştürmek için **Sentence-Transformers** tabanlı **`intfloat/multilingual-e5-small`** mimarisi kullanılmıştır.

### E5 Prefix Kuralı:
E5 modelleri metnin türüne göre başına özel etiketler (prefix) alarak çalışır:
* **Aday CV'si (Sorgu):** Başına `query: ` eklenir.  
  *Örnek:* `query: Ad Soyad: Ahmet Yılmaz Pozisyon: Gaziantep Baklava & Yufka Ustası ...`
* **İş İlanı (Hedef Metin):** Başına `passage: ` eklenir.  
  *Örnek:* `passage: İş İlanı: Gaziantep Usulü Baklava Ustası Firma: Marmara Teknik ...`

Model her iki metni de **384 boyutlu** yoğun birer sayı vektörüne dönüştürür. İki vektör arasındaki açı (Kosinüs Benzerliği - Cosine Similarity) hesaplanarak **Ham Semantik Benzerlik (0.00 - 1.00)** puanı elde edilir.

---

## ⚙️ 3. Hibrit Sıralama (Reranking) Motoru: Çürük Elmalar Nasıl Elenir?

Yalnızca semantik benzerlik yetersizdir. Örneğin bir *"Baklava Ustası"* ile *"Sıva Ustası"* ilanında da "Usta", "çıraklık", "atölye", "mesleki yeterlilik" kelimeleri geçtiği için model ikisini birbirine yakın algılayabilir. 

Bunu önlemek için **Hibrit Filtreleme Katmanı** geliştirilmiştir:

$$\text{Nihai Skor} = \text{Ham Semantik Puan} \times \text{Yaka Çarpanı} \times \text{Kıdem Çarpanı} \times \text{Lokasyon Çarpanı} \times \text{Sektör Çarpanı}$$

### Çarpanların Görevleri:
1. **Yaka Çarpanı (`col_m`):**
   * Aday Mavi Yaka ise Beyaz Yaka ilanlar sıfırla çarpılır (`0.0`), kesinlikle önerilmez.
2. **Kıdem Çarpanı (`sen_m`):**
   * Stajyere Direktör rolü veya 15 yıllık Başmühendis rolüne Giriş Seviyesi ilan önerilmemesi için seviye farkına göre puan kırılır (`0.70` - `1.00`).
3. **Lokasyon Çarpanı (`loc_m`):**
   * İlan `Remote` (Uzaktan) ise ceza uygulanmaz (`1.0`).
   * Aynı şehirde ise tam puan (`1.0`).
   * Şehir farklıysa ceza kesilir (Mavi yakada şehir değiştirme zor olduğu için ceza daha sert uygulanır).
4. **Sektör Çarpanı (`ind_m`):**
   * Adayın sektörü ile ilanın sektörü uyuşuyorsa çarpan: `1.0`
   * Farklıysa (Örn: Gıda ustasına İnşaat veya CNC ustası gelmesi gibi durumlarda): ceza uygulanır (`0.50`). Bu sayede alakasız sektörden sızan ilanlar anında Top-10 listesinden düşer.

---

## 📊 4. Test Sistemi ve Metrikler: Neden F1 Skoru Değil?

Bu model bir "0 veya 1" ikili sınıflandırma modeli **değildir**. Bir arama motoru gibi binlerce ilan arasından en iyi 10 ilanı sırayla getiren bir **Bilgi Getirimi (Information Retrieval / Recommendation)** sistemidir. Bu nedenle F1 skoru yerine aşağıdaki metriklerle test edilir:

* **Top-1 Doğruluğu (%79.78):**
  * Modelin adaya sunduğu **1 numaralı (en tepedeki)** önerinin adayın birebir kendi sektöründen/mesleğinden olma oranıdır.
* **Precision@10 (Öneri Doğruluğu - %67.87):**
  * Modelin listelediği 10 ilanın ortalama kaç tanesinin adayın uzmanlık alanına hitap ettiğini gösterir.
* **NDCG@10 (Normalized Discounted Cumulative Gain - %86.93):**
  * Sıralama kalitesini ölçer. En alakalı ilanın 1. sırada çıkması ile 10. sırada çıkması aynı değildir. Doğru ilanlar en üst sıralara yerleştikçe NDCG %100'e yaklaşır.
* **Filtre Kademeleri:**
  * **Raw:** Filtresiz, yalın yapay zeka puanı.
  * **Collar Filter:** Mavi/Beyaz yaka ayrımı uygulandıktan sonraki başarı.
  * **All Filters:** Şehir, kıdem ve sektör çarpanları uygulandıktan sonraki son kullanıcı deneyimi.

---

## 🗂️ 5. Dosyaların Görev Dağılımı ve Mimarisi

```
konteyner/
├── cv_scoring/yeni_finetune/
│   ├── multilingual-e5-small/           # Fine-tune edilmiş nihai derin öğrenme model ağırlıkları
│   ├── job_embeddings.npy               # 3.000 iş ilanının önceden hesaplanmış 384-boyutlu vektörleri
│   │
│   ├── match_candidate.py              # CANLI EŞLEŞTİRME CLI ARACI:
│   │                                    # Konsoldan CV ID girildiğinde (örn: 10587 veya 20951)
│   │                                    # adaya özel Top-10 ilanı çarpanlarıyla birlikte anında döker.
│   │
│   ├── test.py                          # DEĞERLENDİRME & TEST MOTORU:
│   │                                    # Test kümesindeki tüm adayları tarar; Precision@10,
│   │                                    # Top-1 ve NDCG@10 metriklerini sektör bazında raporlar.
│   │
│   ├── sbertyeni_e5.py                  # MODEL EĞİTİM SCRIPTI:
│   │                                    # Çiftli veri setiyle (train_pairs.csv) E5 modelini
│   │                                    # CosineSimilarityLoss kullanarak eğiten kod.
│   │
│   └── veriseti/
│       ├── tum_cvler.csv                # 3.043 adet zenginleştirilmiş aday CV havuzu (ID: 10000-21043)
│       └── train_pairs.csv              # Pozitif ve negatif eşleşmelerden oluşan model eğitim veri seti
│
├── tum_ilanlar.sql                      # PostgreSQL veritabanındaki 3.000 ilanın şeması ve kayıtları
└── MODEL_VE_TEST_REHBERI.md             # Bu dokümantasyon dosyası
```

---

## 🛠️ 6. Nasıl Çalıştırılır? (Hızlı Başlangıç)

### 1) Bir Aday İçin İlan Önerilerini Görmek (Canlı Eşleştirme):
Konsoldan [`match_candidate.py`](file:///c:/Users/akcic/Desktop/konteyner/cv_scoring/yeni_finetune/match_candidate.py) çalıştırılır:
```powershell
python cv_scoring/yeni_finetune/match_candidate.py
```
* Senden `Aday CV ID` isteyecektir.
* **Örnek Test ID'leri:**
  * `10587`: Uçuş Harekât Uzmanı (Aviation - Beyaz Yaka)
  * `20951`: Gaziantep Baklava & Yufka Ustası (Chef/Horeca - Mavi Yaka)
  * `10042`: Kıdemli Java & Spring Boot Geliştirici (IT - Beyaz Yaka)

### 2) Modelin Genel Test Skorlarını Hesaplamak:
```powershell
python cv_scoring/yeni_finetune/test.py
```
Bu komut test kümesindeki tüm adayları değerlendirip sektör bazlı doğruluk ve NDCG@10 tablosunu basar.

---

## 🔍 7. Önemli Örnek Vaka Analizleri (Edge Cases)

* **Vaka 1: Aday 20951 (Baklava Ustası):**
  * *Başarı:* 1. sırada `0.9049` ham benzerlikle **Gaziantep Usulü Baklava Ustası** ilanı eşleşti.
  * *Karşılaşılan Sorun:* Listenin alt sıralarına Tuğla Ustası ve Sıva Ustası gibi inşaat ilanları sızdı.
  * *Çözüm:* "Usta" kelimesinin sektörler arası ortak olmasından kaynaklanan bu durum, sektör cezasının (`ind_m`) `0.50` seviyesine çekilmesiyle giderildi; inşaat ilanları elenerek Top-10'un tamamı restoran/gıda ilanlarına dönüştürüldü.

* **Vaka 2: Aday 10587 (Uçuş Dispeçeri):**
  * *Başarı:* İlk 3 sırada Pegasus ve SunExpress Dispeçer ilanları tam isabetle önerildi.
  * *Karşılaşılan Sorun:* Listenin 4. sırasına boş açıklamalı bir "Distribütör" ilanı sızdı ("Dispatcher" ile "Distributor" fonetik/subword benzerliği).
  * *Çözüm:* İlan metin temizliği ve sektör çarpanı sayesinde ilgisiz satış ilanları alt sıralara itildi.
