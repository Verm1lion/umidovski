# -*- coding: utf-8 -*-
import sys
import os
import argparse
import json
import sqlalchemy as alc
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Windows konsol UTF-8 çıktı desteği
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMB_CACHE_PATH = os.path.join(BASE_DIR, 'job_embeddings.npy')

# test dosyasından ortak filtreler, sabitler ve yardımcı fonksiyonlar çağrılır
from test import (
    MODEL_PATH,
    clean_text,
    tr_normalize,
    classify_collar,
    extract_cand_title,
    extract_cand_seniority,
    get_cand_city,
    extract_job_city,
    get_seniority_multiplier,
    get_location_multiplier,
    is_industry_match
)

def normalize_collar(c):
    s = str(c).upper().strip()
    if 'BEYAZ' in s or 'WHITE' in s:
        return 'WHITE'
    if 'MAVI' in s or 'BLUE' in s:
        return 'BLUE'
    return 'WHITE'

def load_all_candidates():
    """
    Tüm aday havuzunu yükler (3.043 CV: 2.000 Beyaz Yaka, 1.000 Mavi Yaka, 43 Stajyer).
    """
    primary_paths = [
        os.path.join(BASE_DIR, 'veriseti', 'tum_cvler.csv'),
        os.path.join(BASE_DIR, 'veriseti', 'cv_3000.csv'),
    ]
    fallback_paths = [
        os.path.join(BASE_DIR, 'veriseti', 'test.csv'),
        os.path.join(BASE_DIR, 'veriseti', 'train_cvs.csv')
    ]
    
    for p in primary_paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                if len(df) > 0 and 'cv_id' in df.columns:
                    return df.drop_duplicates(subset=['cv_id']).reset_index(drop=True)
            except Exception:
                pass
                
    dfs = []
    for p in fallback_paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                dfs.append(df)
            except Exception:
                pass
    if not dfs:
        raise FileNotFoundError("Hiçbir aday veriseti dosyası (tum_cvler.csv / test.csv / train_cvs.csv) bulunamadı!")
        
    combined = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['cv_id']).reset_index(drop=True)
    return combined

def process_candidate(cand, df_jobs, job_metadata, job_embeddings, model, top_k=10):
    # 1. Unvan Çıkarımı:
    c_title = str(cand.get('NaturalTitle') or '').strip()
    if not c_title or c_title.lower() == 'nan':
        if 'cv_json' in cand and pd.notnull(cand['cv_json']):
            try:
                c_title = str(json.loads(cand['cv_json']).get('headline', '')).strip()
            except Exception:
                pass
    if not c_title or c_title.lower() == 'nan':
        c_title = extract_cand_title(cand['cv_text'])

    # 2. Sektör Çıkarımı:
    c_ind = str(cand.get('Category') or cand.get('cv_industry') or '').strip()
    if (not c_ind or c_ind.lower() == 'nan') and 'cv_json' in cand and pd.notnull(cand['cv_json']):
        try:
            c_ind = str(json.loads(cand['cv_json']).get('industry', 'UNSPECIFIED')).strip()
        except Exception:
            pass
    if not c_ind or c_ind.lower() == 'nan':
        c_ind = 'UNSPECIFIED'

    # 3. Yaka Çıkarımı (Türkçe/İngilizce normalize edilmiş):
    raw_col = str(cand.get('collar') or '').strip()
    if not raw_col or raw_col.upper() == 'NAN':
        raw_col = classify_collar(c_title)
    c_col = normalize_collar(raw_col)

    # 4. Kıdem Çıkarımı:
    c_sen = extract_cand_seniority(c_title)

    # 5. Lokasyon:
    c_city = str(cand.get('location') or '').strip().upper()
    if not c_city or c_city == 'NAN':
        c_city = get_cand_city(cand)

    c_clean = "query: " + clean_text(cand['cv_text'])

    display_collar = "BEYAZ YAKA (WHITE)" if c_col == 'WHITE' else "MAVİ YAKA (BLUE)"

    print("\n" + "=" * 90)
    print("### SEÇİLEN ADAY PROFİLİ ###")
    print("=" * 90)
    print(f"Aday ID           : {cand['cv_id']}")
    print(f"Mesleki Unvan     : {c_title}")
    print(f"Sektör (Industry) : {c_ind}")
    print(f"Yaka (Collar)     : {display_collar}")
    print(f"Kıdem Seviyesi    : {c_sen}")
    print(f"Lokasyon (Şehir)  : {c_city}")
    print("-" * 90)
    cv_lines = str(cand['cv_text']).strip().split('\n')
    preview = "\n".join(cv_lines[:6])
    print(f"[CV Özeti]:\n{preview}\n...")
    print("-" * 90)

    # Vektör üretimi ve benzerlik hesabı
    c_emb = model.encode([c_clean], normalize_embeddings=True)[0]
    raw_sims = np.dot(job_embeddings, c_emb)

    job_collars = job_metadata['collars']
    job_seniorities = job_metadata['seniorities']
    job_cities = job_metadata['cities']
    job_remotes = job_metadata['remotes']
    job_industries = job_metadata['industries']

    num_jobs = len(df_jobs)
    reranked_sims = raw_sims.copy()
    multipliers = []

    for j in range(num_jobs):
        # Yaka Filtresi: Normalize edilmiş karşılaştırma (WHITE == WHITE, BLUE == BLUE)
        collar_m = 1.0 if c_col == job_collars[j] else 0.0

        # Kıdem ve Lokasyon Çarpanları
        sen_m = get_seniority_multiplier(c_sen, job_seniorities[j])
        loc_m = get_location_multiplier(c_city, job_cities[j], job_remotes[j], c_col)

        # Sektör Çarpanı: Yumuşak filtre
        if c_ind not in ['UNSPECIFIED', 'Bilinmiyor', ''] and not is_industry_match(c_ind, job_industries[j]):
            ind_m = 0.85
        else:
            ind_m = 1.0

        tot_m = collar_m * sen_m * loc_m * ind_m
        reranked_sims[j] = raw_sims[j] * tot_m
        multipliers.append((collar_m, sen_m, loc_m, ind_m, tot_m))

    top_indices = np.argsort(reranked_sims)[::-1][:top_k]

    print("\n" + "=" * 90)
    print(f"### ADAY İÇİN EN UYGUN TOP-{top_k} İLAN VE EŞLEŞME SKORLARI ###")
    print("=" * 90)

    for rank, j_idx in enumerate(top_indices, 1):
        j = df_jobs.iloc[j_idx]
        col_m, sen_m, loc_m, ind_m, tot_m = multipliers[j_idx]
        desc_clean = clean_text(str(j['description']))
        loc_str = j['city'] if j['city'] else 'Belirtilmemiş'
        
        print(f"\n{rank:2d}. [{reranked_sims[j_idx]:.4f}] {j['title']}")
        print(f"    * İlan ID         : {j['id']}")
        print(f"    * Şirket          : {j['company_name']}")
        print(f"    * Sektör          : {j['industry']}")
        print(f"    * Kıdem / Seviye  : {j['position_level']}")
        print(f"    * Çalışma Şekli   : {j['mode_of_operation']}")
        print(f"    * Lokasyon        : {loc_str}")
        print(f"    * Skor Detayı     : Filtreli: {reranked_sims[j_idx]:.4f} | Ham Benzerlik: {raw_sims[j_idx]:.4f}")
        print(f"    * Çarpanlar       : Yaka: {col_m:.2f} | Kıdem: {sen_m:.3f} | Lokasyon: {loc_m:.2f} | Sektör: {ind_m:.2f} => Toplam: {tot_m:.4f}")
        print(f"    * İş Tanımı Özeti : {desc_clean[:300]}...")

    print("\n" + "=" * 90)
    print("Eşleştirme tamamlandı.\n")

def main():
    parser = argparse.ArgumentParser(description="Aday ID'sine göre en uygun ilanları listeler (Döngü Modu).")
    parser.add_argument('cv_id', nargs='?', type=str, help="İlk sorgulanacak Aday ID (Opsiyonel)")
    parser.add_argument('--top_k', type=int, default=10, help="Listelenecek ilan sayısı (Varsayılan: 10)")
    args = parser.parse_args()

    print("\n" + "=" * 90)
    print("ADAY VE İLAN EŞLEŞTİRME SİSTEMİ BAŞLATILIYOR...")
    print("=" * 90)

    print("\n[1/4] Aday veritabanı taranıyor...")
    df_cands = load_all_candidates()
    print(f"Toplam {len(df_cands)} aday kaydı yüklendi (ID Aralığı: {df_cands['cv_id'].min()} - {df_cands['cv_id'].max()}).")

    print("\n[2/4] İş ilanları PostgreSQL veritabanından çekiliyor...")
    engine = alc.create_engine('postgresql://postgres:denemeamacli@localhost:5432/postgres')
    df_jobs = pd.read_sql(
        "SELECT id, title, company_name, description, industry, position_level, mode_of_operation, raw_data FROM job_advertisements ORDER BY id",
        engine
    )
    df_jobs['collar'] = df_jobs['title'].apply(classify_collar).apply(normalize_collar)
    df_jobs['city'] = df_jobs['raw_data'].apply(extract_job_city)
    df_jobs['is_remote'] = df_jobs['mode_of_operation'].astype(str).str.lower() == 'remote'
    
    df_jobs['job_clean_text'] = (
        "passage: İş İlanı: " + df_jobs['title'].astype(str) + 
        " Firma: " + df_jobs['company_name'].fillna('') +
        " Açıklama: " + df_jobs['description'].astype(str).str[:2000]
    ).apply(clean_text)

    # İlan metaverilerini döngü öncesinde tek seferde liste yapısına alıyoruz (hız için)
    job_metadata = {
        'collars': df_jobs['collar'].tolist(),
        'seniorities': df_jobs['position_level'].tolist(),
        'cities': df_jobs['city'].tolist(),
        'remotes': df_jobs['is_remote'].tolist(),
        'industries': df_jobs['industry'].tolist()
    }
    print(f"Toplam {len(df_jobs)} ilan verisi hazırlandı.")

    print(f"\n[3/4] Model yükleniyor ({MODEL_PATH})...")
    model = SentenceTransformer(MODEL_PATH)
    model.max_seq_length = 512

    print("\n[4/4] İlan vektörleri yükleniyor/kontrol ediliyor...")
    if os.path.exists(EMB_CACHE_PATH):
        job_embeddings = np.load(EMB_CACHE_PATH)
        if len(job_embeddings) != len(df_jobs):
            print("İlan sayısı değiştiği için önbellek güncelleniyor...")
            job_embeddings = model.encode(df_jobs['job_clean_text'].tolist(), show_progress_bar=True, normalize_embeddings=True)
            np.save(EMB_CACHE_PATH, job_embeddings)
    else:
        print("İlan vektörleri oluşturuluyor ve kaydediliyor...")
        job_embeddings = model.encode(df_jobs['job_clean_text'].tolist(), show_progress_bar=True, normalize_embeddings=True)
        np.save(EMB_CACHE_PATH, job_embeddings)

    print("\n" + "=" * 90)
    print("SİSTEM HAZIR: Hızlı sorgulama moduna geçildi.")
    print("Çıkış yapmak için '0' veya 'q' tuşuna basıp Enter'a basabilirsiniz.")
    print("=" * 90)

    # CLI üzerinden ilk ID verildiyse önce onu işle
    first_run = True
    initial_id = args.cv_id

    while True:
        if first_run and initial_id is not None:
            user_input = str(initial_id).strip()
            first_run = False
        else:
            first_run = False
            try:
                user_input = input("\nLütfen Aday ID giriniz (Çıkmak için 0): ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\nÇıkış yapıldı. İyi çalışmalar!")
                break

        if not user_input:
            continue

        if user_input in ['0', 'q', 'Q', 'exit', 'quit', 'cikis', 'çıkış']:
            print("\nProgram sonlandırıldı. İyi çalışmalar!")
            break

        match_cands = df_cands[df_cands['cv_id'].astype(str) == user_input]
        if len(match_cands) == 0:
            print(f"\n[UYARI] '{user_input}' ID'sine sahip aday bulunamadı!")
            print("Güncel Geçerli ID Aralıkları:")
            print("  - Beyaz Yaka   : 10000 - 11999 (Örn: 10000, 10001, 10100...)")
            print("  - Mavi Yaka    : 20000 - 20999 (Örn: 20000, 20001, 20150...)")
            print("  - Stajyerler   : 21001 - 21043 (Örn: 21001, 21013, 21034...)")
            continue

        cand = match_cands.iloc[0]
        process_candidate(cand, df_jobs, job_metadata, job_embeddings, model, top_k=args.top_k)
        
        # Eğer CLI parametresi olarak tek bir ID verildiyse ve arka planda/borulamada çalışıyorsa çıkış yap
        if initial_id is not None and not sys.stdin.isatty():
            break

if __name__ == '__main__':
    main()
