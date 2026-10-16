# -*- coding: utf-8 -*-
"""FAZ 2 raporunu (reports/FAZ2_VERI_RAPORU.md) uretilmis JSON ozetlerinden otomatik yazar ve
yerelde (GPU gerektirmeyen) gercek bir referans olcum yapar: yeni sizintisiz test kumesinde BM25.

Girdiler: data_bundle/metadata.json, reports/annotation_iaa.json, reports/annotation_stats.json,
          reports/synthetic_cv_stats.json (ops.), reports/qrels_validation.json (ops.)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, write_json  # noqa: E402

sys.path.insert(0, ROOT)
import match_candidate_sota as core  # noqa: E402

D = os.path.join(ROOT, "data_bundle")
R = os.path.join(ROOT, "reports")


def jl(p):
    p = os.path.join(R, p)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def local_bm25():
    cvs = pd.read_parquet(os.path.join(D, "clean_cvs.parquet"))
    jobs = pd.read_parquet(os.path.join(D, "clean_jobs.parquet")).reset_index(drop=True)
    q = pd.read_parquet(os.path.join(D, "qrels_ground_truth.parquet"))
    jidx = {k: i for i, k in enumerate(jobs.job_key)}
    qrels = {}
    for c, j, g in zip(q.cv_id, q.job_key.map(jidx), q.grade):
        qrels.setdefault(int(c), {})[int(j)] = int(g)
    test = cvs[cvs.split == "test"]
    bm = core.SparseBM25([core.tokenize(t) for t in jobs.job_struct])
    S = bm.scores([core.tokenize(t) for t in test.cv_struct])
    top = core.topk_indices(S, 100)
    ranked = {int(c): top[i].tolist() for i, c in enumerate(test.cv_id)}
    res = core.evaluate_rankings(ranked, {int(c): qrels.get(int(c), {}) for c in test.cv_id}, k=10, rel_threshold=2)
    # eski metrik ile ayni siralama (sektor esitligi + yerel IDCG) -> karsilastirma
    old = {"P@10(sektor)": [], "Top1(sektor)": []}
    ind = jobs.industry.to_numpy(dtype=object)
    for i, c in enumerate(test.cv_id):
        rel = ind[top[i][:10]] == test.gold_industry.iloc[i]
        old["P@10(sektor)"].append(rel.mean())
        old["Top1(sektor)"].append(rel[0])
    s = res["summary"]
    out = {k: round(100 * v, 2) for k, v in s.items() if isinstance(v, float)}
    out.update({k: round(100 * float(np.mean(v)), 2) for k, v in old.items()})
    out["n_queries"] = s["n_queries"]
    out["n_skipped_no_relevant"] = s["n_skipped_no_relevant"]
    lo, hi = core.bootstrap_ci(res["per_query"]["NDCG@10"].to_numpy())
    out["NDCG@10_CI95"] = [round(100 * lo, 1), round(100 * hi, 1)]
    return out


def fmt_tbl(d, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in d:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def main():
    meta = json.load(open(os.path.join(D, "metadata.json"), encoding="utf-8"))
    iaa = jl("annotation_iaa.json") or {}
    ast = jl("annotation_stats.json") or {}
    syn = jl("synthetic_cv_stats.json")
    qv = jl("qrels_validation.json")
    bm = local_bm25()
    write_json(bm, os.path.join(R, "local_bm25_test.json"))
    sp = meta["split"]
    chk = meta["sizinti_kontrolleri"]
    md = []
    md.append("# FAZ 2 — Sızıntısız Veri Mühendisliği, Kademeli Ground Truth ve Zor Negatifler\n")
    md.append("> Bu rapor `scripts/10_make_faz2_report.py` tarafından üretilmiş JSON özetlerinden otomatik yazılır. "
              "Ham sayılar: `data_bundle/metadata.json`, `reports/annotation_*.json`, `reports/synthetic_cv_stats.json`, "
              "`reports/qrels_validation.json`, `reports/local_bm25_test.json`.\n")
    md.append("## 1. Boru hattı\n")
    md.append("```\n01 SQL ayrıştırma ─► 02 anotasyon öğeleri ─► [ISCO-08 anotasyonu, 68 parça] ─► 04 birleştirme + IAA\n"
              "   ─► [konsolidasyon, 43 ISCO grubu] ─► 04 finalize ─► 06 sentetik CV hedefleri ─► [üretim + kör doğrulama]\n"
              "   ─► 06 merge (gidiş-dönüş filtresi) ─► 05 veri paketi (split, qrels, triplet, sızıntı kontrolleri) ─► data_bundle.zip\n```\n")
    md.append("## 2. Meslek anotasyonu (ISCO-08 + uzmanlık + yaka + kıdem)\n")
    md.append(f"* Kılavuz: `annotation/GUIDELINES.md`. 6.043 öğe (3.043 CV + 3.000 ilan), aynı sektör ve benzer unvanlar aynı "
              f"parçada olacak şekilde 68 parça. Ham kodlar `annotation/raw/`.\n")
    md.append(f"* Konsolidasyon sonrası: **{ast.get('isco4_sayisi', '?')} ISCO-08 birim grubu**, "
              f"**{ast.get('occ_key_sayisi', '?')} kanonik uzmanlık**; hem CV hem ilan tarafında bulunan uzmanlık: "
              f"{ast.get('occ_key_hem_cv_hem_ilan', '?')}, ISCO grubu: {ast.get('isco4_hem_cv_hem_ilan', '?')}. "
              f"Konsolidasyonda ISCO kodu değişen öğe: {ast.get('isco4_degisen_oge', '?')}, uzmanlık adı değişen: "
              f"{ast.get('specialty_degisen_oge', '?')}.\n")
    md.append("### 2.1 Anotatörler arası uyum (rastgele %10 alt küme, bağımsız ikinci anotasyon, farklı parça bileşimi)\n")
    rows = [
        ["ISCO-08 4 hane (birim grup)", iaa.get("isco4_kappa"), iaa.get("isco4_birebir_uyum")],
        ["ISCO-08 3 hane", iaa.get("isco3_kappa"), "—"],
        ["ISCO-08 2 hane", iaa.get("isco2_kappa"), "—"],
        ["Yaka (BLUE/WHITE)", iaa.get("collar_kappa"), iaa.get("collar_uyum")],
        ["Kıdem 0–5 (ikinci dereceden ağırlıklı κ)", iaa.get("seniority_quadratic_kappa"), iaa.get("seniority_birebir")],
        ["Veri setindeki CV yakası ↔ anotatör yakası", iaa.get("cv_dataset_collar_vs_anotator_kappa"), iaa.get("cv_dataset_collar_vs_anotator_uyum")],
    ]
    md.append(f"n = {iaa.get('n', '?')} öğe.\n\n" + fmt_tbl(rows, ["Alan", "Cohen κ", "Birebir uyum"]) + "\n")
    md.append(f"\nKıdemde |fark| ≤ 1 oranı: {iaa.get('seniority_fark_le1')}. Landis & Koch (1977) ölçeğinde κ > 0,80 "
              "'neredeyse mükemmel' uyumdur. Uzmanlık adlarının konsolidasyon öncesi birebir yazım uyumu "
              f"{iaa.get('specialty_birebir_(konsolidasyon_oncesi)')}'dir; bu yüzden ad farklılıkları ayrı bir "
              "konsolidasyon turunda kanonikleştirilmiştir (ISCO grubu düzeyinde).\n")
    md.append("\n**Yaka (gold):** Yaka mesleğin bir özelliğidir; aynı uzmanlıktaki tüm öğeler için çoğunluk oyu "
              "(CV + ilan birlikte) kullanılır. Veri setindeki `collar` alanı ile uyum yukarıdadır.\n")
    md.append("\n**Anotatörlerin veri kalitesi notları:** Birçok sentetik CV'de deneyim/eğitim metni unvanla "
              "uyuşmayan şablon gürültüsü taşıyor (ör. astrofizik unvanlı CV'de bulut/veri deneyimi). Bu CV'ler "
              "unvan + özet + becerilerden kodlandı ve `conf` düşürüldü. Güven dağılımı: "
              f"{ast.get('conf_dagilimi')}.\n")
    md.append("## 3. Kademeli alaka (graded relevance) kuralları\n")
    md.append(fmt_tbl([[k, v] for k, v in sorted(meta["alaka_dereceleri"].items(), reverse=True)], ["Grade", "Kural"]) + "\n")
    md.append(f"\nqrels (grade ≥ 1): **{meta['qrels']['satir(grade>=1)']:,}** satır; dağılım {meta['qrels']['grade_dagilimi']}. "
              "Listede olmayan tüm çiftler grade 0'dır. **Tüm** CV × ilan çiftleri yargılıdır (tam yargı).\n")
    if qv:
        md.append("### 3.1 Kural-türevi derecelerin bağımsız çift-düzeyi yargılarla doğrulanması\n")
        md.append(f"Tabakalı örneklem (her dereceden eşit sayıda, yalnız gerçek val/test CV'leri), n = {qv['n']}. Yargıçlar "
                  "kural derecesini görmeden yalnızca CV ve ilan metnini okuyup 0–3 verdi "
                  "(`annotation/qrels_validation/RUBRIC.txt`).\n\n")
        md.append(fmt_tbl([["İkinci dereceden ağırlıklı κ", qv["quadratic_weighted_kappa"]],
                           ["Doğrusal ağırlıklı κ", qv["linear_weighted_kappa"]],
                           ["Birebir uyum", qv["birebir_uyum"]], ["|fark| ≤ 1", qv["fark_le1"]],
                           ["İkili (grade ≥ 2) uyum / κ", f"{qv['ikili_uyum_(>=2_alakali)']} / {qv['ikili_kappa_(>=2)']}"],
                           ["Spearman ρ", qv["spearman"]]], ["Ölçüt", "Değer"]) + "\n")
        cm = qv["karisiklik_matrisi_satir=kural_sutun=yargi"]
        md.append("\nKarışıklık matrisi (satır = kural, sütun = bağımsız yargı):\n\n" +
                  fmt_tbl([[g] + cm[g] for g in range(4)], ["kural \\ yargı", "0", "1", "2", "3"]) + "\n")
    md.append("## 4. Sızıntısız bölme\n")
    rows = [[s, sp[s]["cv"], sp[s]["gercek"], sp[s]["sentetik"], sp[s]["ort_grade3"], sp[s]["ort_grade2"],
             sp[s]["grade3_ilani_olan_cv_orani"], sp[s]["alakali(>=2)_ilani_olmayan_cv"], sp[s]["alakali_sayisi_medyan"]]
            for s in ("train", "val", "test")]
    md.append(fmt_tbl(rows, ["Bölüm", "CV", "gerçek", "sentetik", "ort. grade-3 ilan", "ort. grade-2 ilan",
                             "grade-3'ü olan CV oranı", "alakalısı olmayan CV", "medyan alakalı (≥2)"]) + "\n")
    md.append(f"\nİlan havuzu: {meta['ilan_havuzu']} (cold = hiçbir eğitim sinyalinde kullanılmaz). Vitrin adayları "
              f"(test'e sabit): {meta['vitrin_adaylari_test']}.\n")
    md.append("### 4.1 Otomatik sızıntı kontrolleri (hepsi `05_build_dataset.py` içinde zorunlu; biri bile başarısızsa paket üretilmez)\n")
    md.append(fmt_tbl([[k, v] for k, v in chk.items()], ["Kontrol", "Değer"]) + "\n")
    md.append("## 5. Zor negatifler ve triplet'ler\n")
    t = meta["triplet"]
    md.append(f"* Train triplet: **{t['train']:,}**, val triplet (yalnız val kaybı için): {t['val']:,}. İstatistik: {t['train_istatistik']}.\n")
    for k, v in t["negatif_turleri"].items():
        md.append(f"* `{k}`: {v}\n")
    md.append(f"* Negatiflerin derece dağılımı: {t['negatif_grade_dagilimi']}\n")
    md.append("* Notebook ayrıca 4. bir negatif üretir: eğitimsiz base modelin üst sıralara koyduğu **grade-0** ilanlar "
              "(qrels ile filtrelendiği için yanlış negatif yok).\n")
    if syn:
        md.append("## 6. Sentetik CV zenginleştirmesi (yalnız train)\n")
        md.append(f"* Hedef: train'de hiç / tek CV'si olmayan ve WARM ilanlarda geçen uzmanlıklar. Üretilen: {syn.get('hedef')}, "
                  f"kabul edilen: **{syn.get('kabul')}**.\n")
        md.append(f"* Gidiş-dönüş (kör anotatör) tutarlılığı: ISCO {syn.get('isco4_tutarli')}, yaka {syn.get('yaka_tutarli')}, "
                  f"kıdem(±1) {syn.get('kidem_tutarli')}; filtre reddi {syn.get('tutarlilik_filtresi_ret', 0)}, "
                  f"val/test başlık çakışması reddi {syn.get('val_test_baslik_cakismasi_ret', 0)}.\n")
    md.append("## 7. Vitrin adayları için kural dereceleri (kontrol)\n")
    for cid, v in meta["vitrin_kontrol"].items():
        md.append(f"**{cid} — {v['headline']}** (split={v['split']}, uzmanlık: {v['gold_specialty']}, ISCO {v['gold_isco4']}, "
                  f"kıdem {v['gold_seniority']}; grade-3 ilan: {v['n_grade3']}, grade-2: {v['n_grade2']})\n")
        md.append("* Alakalı örnekler: " + "; ".join(v["ornek_alakali"][:6]) + "\n")
        traps = "; ".join(f"{k} → {g}" for k, g in list(v["tuzak_ilan_dereceleri"].items())[:8])
        md.append(f"* Tuzak ilanlar ve dereceleri: {traps}\n\n")
    md.append("## 8. Yeni test kümesinde yerel referans ölçüm: BM25 (GPU gerekmez, gerçek sayı)\n")
    md.append(fmt_tbl([[k, v] for k, v in bm.items()], ["Metrik", "BM25 (test)"]) + "\n")
    md.append("\nAynı BM25 sıralaması eski metrikle (\"aynı sektör\") ölçülünce çok daha yüksek görünür. Yeni kademeli "
              "qrels meslek, yaka, kıdem ve lokasyonu ölçtüğü için çok daha katıdır. Nöral sistemlerin sayıları Colab "
              "A100 çalıştırmasıyla notebook'ta üretilir.\n")
    md.append("## 9. Paket içeriği\n")
    md.append(fmt_tbl([[f, f"{v['bayt'] / 1e6:.2f} MB", v["sha256"][:16] + "…"] for f, v in meta["dosyalar"].items()],
                      ["Dosya", "Boyut", "SHA-256"]) + "\n")
    with open(os.path.join(R, "FAZ2_VERI_RAPORU.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("-> reports/FAZ2_VERI_RAPORU.md")
    print(json.dumps(bm, ensure_ascii=False))


if __name__ == "__main__":
    main()
