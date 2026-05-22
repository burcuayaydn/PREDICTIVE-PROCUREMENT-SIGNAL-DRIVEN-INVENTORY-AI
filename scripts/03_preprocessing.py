# ==============================================================================
# PROJE: PREDICTIVE PROCUREMENT & SIGNAL-DRIVEN INVENTORY AI
# MODÜL: 03_PREPROCESSING.PY
# ==============================================================================

import pandas as pd
import numpy as np
import os


def run_preprocessing():
    print("Sistem: Kesintisiz Zaman Matrisi (Time Grid) ve Feature Engineering Motoru tetiklendi...")

    # --- GARANTİLİ KÖK DİZİN BULMA MEKANİZMASI ---
    current_dir = os.path.dirname(os.path.abspath(__file__))

    if "RetailRocket" in current_dir:
        base_path = current_dir.split("RetailRocket")[0] + "RetailRocket"
    else:
        base_path = os.path.dirname(current_dir)  # Bir üst klasöre çık (scripts'ten köke)

    input_path = os.path.normpath(os.path.join(base_path, "data", "processed", "master_data.parquet"))
    output_path = os.path.normpath(os.path.join(base_path, "data", "processed", "engineered_features.parquet"))

    print(f"Sistem Kontrolü -> Girdi dosyası aranıyor:\n  {input_path}")
    print(f"Sistem Kontrolü -> Çıktı dosyası buraya yazılacak:\n  {output_path}")

    if not os.path.exists(input_path):
        alternative_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        input_path = os.path.normpath(os.path.join(alternative_base, "data", "processed", "master_data.parquet"))
        output_path = os.path.normpath(
            os.path.join(alternative_base, "data", "processed", "engineered_features.parquet"))

        if not os.path.exists(input_path):
            raise FileNotFoundError(
                f"⚠️ HATA: 'cleaned_events.parquet' bulunamadı!\n"
                f"Lütfen Stage 02 kodunun çıktıyı TAM OLARAK nereye kaydettiğini kontrol edin.\n"
                f"Aranan Konum: {input_path}"
            )

    df = pd.read_parquet(input_path)

    # Tarihi haftalık periyotların başlangıcına yuvarlıyoruz
    df['week_start'] = df['date'].dt.to_period('W').dt.to_timestamp()

    print("Sistem: Haftalık bazda aksiyon sayıları (View, Cart, Transaction) pivot ediliyor...")
    weekly_pivot = df.groupby(['item_id', 'week_start', 'event_type']).size().unstack(fill_value=0).reset_index()
    weekly_pivot.rename(
        columns={'week_start': 'date', 'view': 'view_count', 'addtocart': 'cart_count', 'transaction': 'sales_count'},
        inplace=True
    )

    # ==============================================================================
    # 🌟 TIME GRID (ZAMAN BOŞLUKLARININ KAPATILMASI)
    # ==============================================================================
    print("Sistem: Seyrek veriyi engellemek için kesintisiz zaman ekseni inşa ediliyor...")
    all_dates = sorted(weekly_pivot['date'].unique())
    all_items = weekly_pivot['item_id'].unique()

    full_index = pd.MultiIndex.from_product([all_items, all_dates], names=['item_id', 'date'])
    weekly_grid = weekly_pivot.set_index(['item_id', 'date']).reindex(full_index, fill_value=0).reset_index()

    weekly_grid = weekly_grid.sort_values(by=['item_id', 'date']).reset_index(drop=True)

    # ==============================================================================
    # 🛠️ FEATURE ENGINEERING (LAG & ROLLING FEATURES)
    # ==============================================================================
    print("Sistem: Kusursuz zaman ekseninde Lag ve Rolling özellikleri üretiliyor...")

    weekly_grid['sales_lag_1'] = weekly_grid.groupby('item_id')['sales_count'].shift(1).fillna(0)
    weekly_grid['cart_lag_1'] = weekly_grid.groupby('item_id')['cart_count'].shift(1).fillna(0)
    weekly_grid['view_lag_1'] = weekly_grid.groupby('item_id')['view_count'].shift(1).fillna(0)

    weekly_grid['sales_roll_mean_3'] = weekly_grid.groupby('item_id')['sales_count'].shift(1).rolling(
        window=3).mean().fillna(0)

    # Boyut Optimizasyonu
    final_df = weekly_grid[
        (weekly_grid['sales_count'] > 0) |
        (weekly_grid['sales_lag_1'] > 0) |
        (weekly_grid['view_lag_1'] > 0)
        ].reset_index(drop=True)

    # Kayıt Adımı
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_parquet(output_path, index=False)
    print(f"✔ BAŞARILI: Veri önişleme ve zaman matrisi kalibrasyonu tamamlandı.\n")


if __name__ == "__main__":
    run_preprocessing()