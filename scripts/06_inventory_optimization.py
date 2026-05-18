# ==============================================================================
# PROJE: PREDICTIVE PROCUREMENT & SIGNAL-DRIVEN INVENTORY AI
# MODÜL: 06_INVENTORY_OPTIMIZATION.PY
# ==============================================================================

import pandas as pd
import numpy as np
import os

# Pandas konsol gösterim ayarları
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def calculate_inventory_metrics_pro():
    print("Sistem: Yapay Zeka Hata Payı (RMSE) & Dinamik Lojistik Motoru tetiklendi...")

    # 1. Klasör Yolları Kurulumu (Garantili Yapı)
    script_dir = os.path.dirname(os.path.abspath(__file__))  # scripts klasörü
    base_path = os.path.dirname(script_dir)  # Proje Kök Klasörü

    os.makedirs(os.path.join(base_path, "data", "processed"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "outputs", "reports"), exist_ok=True)

    input_path = os.path.normpath(os.path.join(base_path, "data", "processed", "model_predictions.parquet"))
    output_path = os.path.normpath(os.path.join(base_path, "data", "processed", "inventory_decisions.parquet"))
    insight_report_path = os.path.normpath(os.path.join(base_path, "outputs", "reports", "business_insights.txt"))

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"⚠️ HATA: Tahmin dosyası bulunamadı! Lütfen önce STAGE 05'i çalıştırın: {input_path}")

    df = pd.read_parquet(input_path)

    # ==============================================================================
    # 🌟 2. STRATEJİK VE DİNAMİK LOJİSTİK PARAMETRELER
    # ==============================================================================
    print("Sistem: Tedarik zinciri risk haritasına göre dinamik değişkenler atanıyor...")

    # [DİNAMİK LEAD TIME]
    # Hızlı akan AX/AY lokomotifleri için öncelikli/hızlı hat (1 Hafta).
    # Normal ürünler için 2 Hafta, yavaş akan ve risk taşımayan CZ ürünleri için maliyet odaklı ucuz hat (3 Hafta)
    df['lead_time_weeks'] = np.where(df['abc_xyz_rank'].isin(['AX', 'AY']), 1, 2)
    df.loc[df['abc_xyz_rank'] == 'CZ', 'lead_time_weeks'] = 3

    # [DİNAMİK HİZMET SEVİYESİ (Z-SCORE)]
    # Sitenin cirosunu sırtlayan AX ve AY gruplarında stoksuz kalma riski %1 (Z = 2.33)
    df['z_score'] = np.where(df['abc_xyz_rank'].isin(['AX', 'AY']), 2.33, 1.645)

    # ==============================================================================
    # 🌟 3. YAPAY ZEKA TAHMİN HATASININ (RMSE) ENVANTER FORMÜLÜNE GÖMÜLMESİ
    # ==============================================================================
    print("Sistem: Emniyet stokları tarihsel varyans yerine yapay zeka hata payı (RMSE) ile kalibre ediliyor...")

    # Her satır için modelin anlık hata payını buluyoruz
    df['forecast_error'] = df['actual_demand'] - df['pred_demand']

    # Segment bazlı tahmin hatası standart sapmasını (RMSE yaklaşımı) hesaplayıp ürüne haritalıyoruz
    df['segment_rmse'] = df.groupby('abc_xyz_rank')['forecast_error'].transform('std').fillna(0.1)

    # FORMÜL: Safety Stock = Z-Score * sqrt(Dinamik Lead Time) * Segment RMSE
    df['safety_stock'] = df['z_score'] * np.sqrt(df['lead_time_weeks']) * df['segment_rmse']
    df['safety_stock'] = np.ceil(df['safety_stock']).astype(int)

    # ==============================================================================
    # 🛠️ 4. REORDER POINT (ROP) VE SİPARİŞ MİKTARI HESABI
    # ==============================================================================
    print("Sistem: Yeniden Sipariş Noktaları (ROP) mühürleniyor...")
    # FORMÜL: ROP = (Tahmini Talep * Dinamik Kurşun Zamanı) + Emniyet Stoku
    df['reorder_point'] = (df['pred_demand'] * df['lead_time_weeks']) + df['safety_stock']
    df['reorder_point'] = np.ceil(df['reorder_point']).astype(int)

    # ÖNERİLEN SİPARİŞ MİKTARI: Haftalık tahmini talebin 4 katı (1 aylık döngü güvenliği)
    df['suggested_order_qty'] = np.ceil(df['pred_demand'] * 4).astype(int)

    # Sonuçların kaydedilmesi
    df.to_parquet(output_path, index=False)
    print(f"✔ BAŞARILI: Lojistik kararlar kaydedildi -> {output_path}")

    # ==============================================================================
    # 🌟 5. STRATEJİK GÜVENLİ RAPORLAMA KATMANI (APPEND MODU)
    # ==============================================================================
    print("Sistem: Yapay zeka destekli lojistik analizler kurumsal rapora ekleniyor...")

    total_items = df['item_id'].nunique()
    avg_safety = df['safety_stock'].mean()
    avg_rop = df['reorder_point'].mean()
    strategic_protected = df[df['abc_xyz_rank'].isin(['AX', 'AY'])]['item_id'].nunique()

    latest_date = df['date'].max()
    df_latest = df[df['date'] == latest_date]

    # Segment bazlı özet istatistikler
    segment_summary = df_latest.groupby('abc_xyz_rank', observed=False).agg(
        toplam_urun=('item_id', 'count'),
        ort_tahmini_talep=('pred_demand', 'mean'),
        ort_emniyet_stoku=('safety_stock', 'mean'),
        ort_rop=('reorder_point', 'mean'),
        dinamik_lead_time=('lead_time_weeks', 'first')
    ).reset_index()

    segment_details_str = ""
    for idx, row in segment_summary.iterrows():
        if row['toplam_urun'] > 0:
            segment_details_str += f"""   -> Envanter Kodu: {row['abc_xyz_rank']}
      - Aktif SKU Sayısı            : {row['toplam_urun']:,} Adet
      - SKU Başına Ort. Haftalık Talep: {row['ort_tahmini_talep']:.3f} Adet
      - Önerilen Güvenlik Rezervi    : {row['ort_emniyet_stoku']:.2f} Adet (RMSE Odaklı)
      - Ortalama ROP Sipariş Sınırı   : {row['ort_rop']:.2f} Adet
      - Atanan Dinamik Tedarik Süresi : {int(row['dinamik_lead_time'])} Hafta
   ---------------------------------------------------------\n"""

    logistics_insight_append = f"""
==================================================================
        STAGE 06: INVENTORY LOGIC & STRATEGIC INSIGHTS (V2 - PRO)
==================================================================
* Analiz Edilen Toplam Benzersiz SKU (Ürün) : {total_items:,}
* Kritik Seviyede Korunan Stratejik Ürün     : {strategic_protected} (AX ve AY Lokomotifleri)
  (Not: Bu stratejik lokomotiflerin stoksuz kalma riski %1'e indirilmiştir. Z-Score: 2.33)
* Sistem Geneli Ortalama Emniyet Stoku     : {avg_safety:.2f} Adet
* Sistem Geneli Ortalama Sipariş Noktası    : {avg_rop:.2f} Adet

[STRATEJİK SEGMENT BAZLI DERİN LOJİSTİK ÖZETİ]
------------------------------------------------------------------
{segment_details_str}
[STRATEJİK YÖNETİM TAVSİYELERİ (EXECUTIVE ACTION PLAN)]
1. Modelimiz, sadece geçmiş satışa bakmak yerine web sitesindeki müşteri klik hunisini (view/cart)
   başarıyla entegre ederek tedarik zincirini tamamen veri güdümlü ve proaktif hale getirmiştir.

2. Emniyet stokları klasik tarihsel dalgalanmalar yerine, makine öğrenmesi modelimizin segment
   bazındaki "tahmin hatası (RMSE)" payına endekslenmiştir. Bu sayede modelin kararlı tahmin
   ürettiği kalemlerde gereksiz stok maliyetleri elenmiş, riskli kalemlerde ise koruma artırılmıştır.

3. AX ve AY segmentlerinde dinamik kargo ve hızlı tedarik (1 Hafta) simüle edilerek çeviklik 
   maksimuma çıkarılmış; CZ segmentindeki yavaş akan "Long-Tail" ürünlerde ise Just-In-Time 
   yaklaşımı ile deponun atıl sermayeye boğulması engellenmiştir.

==================================================================
"""

    with open(insight_report_path, "a", encoding="utf-8") as f:
        f.write(logistics_insight_append)

    print(f"✔ BAŞARILI: Endüstriyel lojistik rapor, 'business_insights.txt' dosyasının sonuna eklendi.")


if __name__ == "__main__":
    calculate_inventory_metrics_pro()