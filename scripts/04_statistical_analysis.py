# ==============================================================================
# PROJE: PREDICTIVE PROCUREMENT & SIGNAL-DRIVEN INVENTORY AI
# MODÜL: 04_STATISTICAL_ANALYSIS.PY (ABC/XYZ RETAIL SEGMENTATION)
# ==============================================================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Grafik Görsel ve Estetik Düzenlemeleri
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]

# Dinamik Klasör Yapısı Ayarları
script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.dirname(script_dir)

input_path = os.path.join(base_path, "data", "processed", "engineered_features.parquet")
output_path = os.path.join(base_path, "data", "processed", "segmented_products.parquet")
figure_path = os.path.join(base_path, "outputs", "figures")
report_path = os.path.join(base_path, "outputs", "reports", "business_insights.txt")


def run_statistical_segmentation():
    print("Sistem: İstatistiki analiz ve ABC/XYZ segmentasyonu başlatılıyor...")
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"HATA: Haftalık zaman serisi verisi bulunamadı! Önce '03_preprocessing.py' çalıştırılmalı.")

    df = pd.read_parquet(input_path)

    # 1. ÜRÜN BAZLI TEMEL İSTATİSTİKLER
    print("Sistem: Ürün bazlı talep kararsızlığı (Varyasyon Katsayısı) hesaplanıyor...")
    product_stats = df.groupby('item_id').agg(
        total_sales=('sales_count', 'sum'),
        total_cart=('cart_count', 'sum'),
        mean_sales=('sales_count', 'mean'),
        std_sales=('sales_count', 'std')
    ).reset_index()

    # Varyasyon Katsayısı (Coefficient of Variation - CV) = Standart Sapma / Ortalama
    # Talebin ne kadar kararsız veya düzenli olduğunu söyler.
    product_stats['cv_sales'] = product_stats['std_sales'] / product_stats['mean_sales']
    # Satışı sıfır veya çok seyrek olan, sapması hesaplanamayan ürünlere yüksek kararsızlık cezası veriyoruz
    product_stats['cv_sales'] = product_stats['cv_sales'].fillna(2.0)

    # 2. ABC ANALİZİ (Toplam Satış Hacmine Göre Önceliklendirme)
    print("Sistem: Kümülatif hacim kırılımıyla ABC analizi yapılıyor...")
    product_stats = product_stats.sort_values(by='total_sales', ascending=False).reset_index(drop=True)
    product_stats['cum_sales'] = product_stats['total_sales'].cumsum()
    total_volume = product_stats['total_sales'].sum()

    if total_volume > 0:
        product_stats['cum_percentage'] = (product_stats['cum_sales'] / total_volume) * 100
    else:
        product_stats['cum_percentage'] = 100.0

    def abc_classify(percentage):
        if percentage <= 70:
            return 'A'  # Satışların ana omurgasını oluşturan lokomotif %70
        elif percentage <= 90:
            return 'B'  # Orta hareketli kademe %20
        return 'C'  # Çok yavaş akan veya kuyrukta kalan %10

    product_stats['abc_class'] = product_stats['cum_percentage'].apply(abc_classify)

    # 3. XYZ ANALİZİ (Tahmin Edilebilirlik ve Envanter Riski Ölçümü)
    print("Sistem: Tahmin edilebilirlik kırılımıyla XYZ analizi yapılıyor...")

    def xyz_classify(cv):
        if cv <= 0.5:
            return 'X'  # Düzenli/Sabit talep (En kolay grup, emniyet stoku düşük tutulabilir)
        elif cv <= 1.2:
            return 'Y'  # Trendli/Dönemsel talep (Yapay zekanın tam kalbi)
        return 'Z'  # Kaotik/Belirsiz talep (Satın alması en riskli grup)

    product_stats['xyz_class'] = product_stats['cv_sales'].apply(xyz_classify)
    product_stats['abc_xyz_rank'] = product_stats['abc_class'] + product_stats['xyz_class']

    # 4. GRAFİK: ABC/XYZ JÜRİ MATRİSİ (HEATMAP)
    print("Sistem: Kurumsal Satın Alma Matrisi grafiğe basılıyor...")
    matrix_data = product_stats.groupby(['abc_class', 'xyz_class']).size().unstack(fill_value=0)

    plt.figure(figsize=(10, 6))
    sns.heatmap(matrix_data, annot=True, fmt="d", cmap="YlGnBu", cbar=False, linewidths=1)
    plt.title("Predictive Procurement - Envanter Dağılım Matrisi (ABC/XYZ SKU Counts)", fontsize=14, fontweight='bold',
              pad=15)
    plt.xlabel("XYZ Sınıfı: Talep Tahmin Edilebilirlik Derecesi (X=Kolay -> Z=Zor)")
    plt.ylabel("ABC Sınıfı: Haftalık Operasyonel Satış Hacmi (A=Yüksek -> C=Düşük)")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_path, "04_abc_xyz_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 5. RAPORA EKLEME YAPMA (APPEND MODU)
    rank_counts = product_stats['abc_xyz_rank'].value_counts()

    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n\n" + "=" * 65 + "\n")
        f.write("    STAGE 04: STRATEJİK ABC/XYZ SEGMENTASYON ÇIKTILARI \n")
        f.write("=" * 65 + "\n")
        f.write(f"Raporlama Zamanı: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("--- 1. ENVENTER KODU VE SKU DAĞILIMI ---\n")
        f.write(f"• Toplam Sınıflandırılan SKU Sayısı : {product_stats.shape[0]:,} benzersiz ürün\n")
        f.write(f"• AX Grubu (Yüksek Hacim & Düzenli)  : {rank_counts.get('AX', 0)} SKU\n")
        f.write(f"• AY Grubu (Yüksek Hacim & Trendli)  : {rank_counts.get('AY', 0)} SKU\n")
        f.write(f"• CZ Grubu (Düşük Hacim & Kaotik)    : {rank_counts.get('CZ', 0)} SKU\n\n")
        f.write("👉 ENVANTER GÜVENLİĞİ VE TEDARİK STRATEJİSİ:\n")
        f.write("   - AX ve AY segmentindeki ürünler web sitesinin ciro motorudur. Bu ürünlerde\n")
        f.write("     stoksuz kalma (stockout) maliyeti çok yüksektir, yapay zeka öncelikli izlemelidir.\n")
        f.write("   - CZ grubu ürünler ise tedarikçiden toplu değil, sipariş geldikçe (Just-In-Time)\n")
        f.write("     veya minimum miktarda geçilmelidir, aksi takdirde depoda ölü stok riski doğar.\n")
        f.write("=================================================================\n")

    # Ana zaman serisi tablosuna bu segmentasyon etiketlerini ekleyip paketliyoruz
    final_df = pd.merge(df, product_stats[['item_id', 'abc_class', 'xyz_class', 'abc_xyz_rank']], on='item_id',
                        how='left')
    final_df.to_parquet(output_path, index=False)

    print(f"\n✔ BAŞARILI: ABC/XYZ Analizi tamamlandı.")
    print(f"✔ Yeni zenginleştirilmiş veri seti '{output_path}' olarak kaydedildi.")


if __name__ == "__main__":
    run_statistical_segmentation()