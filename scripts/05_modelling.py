# ==============================================================================
# PROJE: PREDICTIVE PROCUREMENT & SIGNAL-DRIVEN INVENTORY AI
# MODÜL: 05_MODELLING.PY
# ==============================================================================

import pandas as pd
import numpy as np
import os
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("Sistem: Üçlü Model Yarıştırma ve En İyi R2 Seçim Motoru Başlatılıyor...")

# Dinamik Klasör Yapısı Ayarları
script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.dirname(script_dir)

input_path = os.path.join(base_path, "data", "processed", "segmented_products.parquet")
output_path = os.path.join(base_path, "data", "processed", "model_predictions.parquet")
report_path = os.path.join(base_path, "outputs", "reports", "business_insights.txt")

if not os.path.exists(input_path):
    raise FileNotFoundError(f"HATA: Segmenter veri bulunamadı! Önce '04_statistical_analysis.py' çalıştırılmalı.")

# 1. VERİ SETİNİN YÜKLENMESİ VE DERİN ÖZELLİK MÜHENDİSLİĞİ
df = pd.read_parquet(input_path)
df['date'] = pd.to_datetime(df['date'])

print("Sistem: Gelişmiş zaman serisi özellikleri (Lag & Rolling) kontrol ediliyor...")
df = df.sort_values(by=['item_id', 'date']).reset_index(drop=True)

# Ekstra Gecikme (Lag) Özellikleri (Güvenli atama)
if 'sales_lag_2' not in df.columns:
    df['sales_lag_2'] = df.groupby('item_id')['sales_count'].shift(2).fillna(0)
if 'sales_lag_3' not in df.columns:
    df['sales_lag_3'] = df.groupby('item_id')['sales_count'].shift(3).fillna(0)

# Kategorik özellikleri model formatlarına hazırlama
categorical_features = ['abc_class', 'xyz_class', 'abc_xyz_rank']
df_encoded = df.copy()

# Ağaç bazlı modeller için kategorik verileri numerik kodlara çeviriyoruz (XGBoost ve RF uyumluluğu için)
for col in categorical_features:
    df_encoded[col] = df_encoded[col].astype('category').cat.codes

# 2. ZAMANSAL KRONOLOJİK SPLIT (Son 2 Hafta Test)
max_date = df_encoded['date'].max()
split_date = max_date - pd.Timedelta(weeks=2)

train_df = df_encoded[df_encoded['date'] < split_date]
test_df = df_encoded[df_encoded['date'] >= split_date]

features = [
    'view_count', 'cart_count',
    'sales_lag_1', 'sales_lag_2', 'sales_lag_3',
    'cart_lag_1', 'view_lag_1',
    'sales_roll_mean_3',
    'abc_class', 'xyz_class', 'abc_xyz_rank'
]
target = 'sales_count'

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

print(f"-> [Eğitim Kümesi Hacmi]: {X_train.shape[0]:,} satır")
print(f"-> [Test Kümesi Hacmi]  : {X_test.shape[0]:,} satır\n")

# 3. MODELLERİN TANIMLANMASI
models = {
    "LightGBM": LGBMRegressor(
        n_estimators=300,
        learning_rate=0.04,
        num_leaves=63,
        max_depth=8,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=50,
        max_depth=12,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
}

# Modelleri yarıştıracağımız metrik tablosu
performance_metrics = {}
saved_predictions = {}

# 4. MODELLERİN EĞİTİLMESİ VE DEĞERLENDİRİLMESİ
for name, model in models.items():
    print(f"Sistem: {name} modeli eğitiliyor...")

    if name == "LightGBM":
        # LightGBM için verileri açıkça 'category' tipine alıyoruz
        X_train_model = X_train.copy()
        X_test_model = X_test.copy()
        df_predict_model = df_encoded[features].copy()

        for col in categorical_features:
            X_train_model[col] = X_train_model[col].astype('category')
            X_test_model[col] = X_test_model[col].astype('category')
            df_predict_model[col] = df_predict_model[col].astype('category')

        model.fit(X_train_model, y_train, eval_set=[(X_test_model, y_test)])
        preds = model.predict(df_predict_model)

    else:
        # XGBoost and Random Forest için kategorileri numerik kodlara (int) çeviriyoruz
        X_train_model = X_train.copy()
        X_test_model = X_test.copy()
        df_predict_model = df_encoded[features].copy()

        for col in categorical_features:
            X_train_model[col] = X_train_model[col].astype('category').cat.codes
            X_test_model[col] = X_test_model[col].astype('category').cat.codes
            df_predict_model[col] = df_predict_model[col].astype('category').cat.codes

        model.fit(X_train_model, y_train)
        preds = model.predict(df_predict_model)

    # Tahminleri sıfırın altına düşmeyecek şekilde kırpıyoruz
    preds = np.clip(preds, 0, None)
    saved_predictions[name] = preds

    # Sadece test seti performansını hesaplama
    test_preds = preds[df_encoded['date'] >= split_date]
    rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    mae = mean_absolute_error(y_test, test_preds)
    r2 = r2_score(y_test, test_preds)

    performance_metrics[name] = {"R2": r2, "RMSE": rmse, "MAE": mae}
    print(f"-> {name} Performansı: R2 = {r2:.4f} | RMSE = {rmse:.4f} | MAE = {mae:.4f}\n")

# ==============================================================================
# EK ADIM: EN BAŞARILI MODELİN SEÇİLMESİ VE RAPORA (BUSINESS_INSIGHTS) YAZILMASI
# ==============================================================================

# R2 skoruna göre otomatik olarak şampiyon modeli belirliyoruz
best_model_name = max(performance_metrics, key=lambda k: performance_metrics[k]["R2"])
best_r2 = performance_metrics[best_model_name]["R2"]

print("=" * 50)
print(f"🏆 YARIŞMA ŞAMPİYONU: {best_model_name} (R2: {best_r2:.4f})")
print("=" * 50)

# Nihai tahminleri Streamlit'in okuyabilmesi için şampiyon modelin çıktılarıyla eşliyoruz
df['pred_demand'] = saved_predictions[best_model_name]

# business_insights.txt dosyasına APPEND (Ekleme) modunda liderlik tablosunu yazıyoruz
with open(report_path, "a", encoding="utf-8") as f:
    f.write("\n\n" + "=" * 70 + "\n")
    f.write("    STAGE 05: YAPAY ZEKA MODEL KARŞILAŞTIRMA VE LİDERLİK TABLOSU \n")
    f.write("=" * 70 + "\n")
    f.write(f"Raporlama Zamanı: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # Hizalı tablo başlıkları
    f.write(f"{'Algoritma Modeli':<20} | {'R2 Skoru':<10} | {'RMSE Hatası':<12} | {'MAE Hatası':<10}\n")
    f.write("-" * 65 + "\n")

    # Döngüyle tüm modellerin performansını alt alta tabloya ekleme
    for name, metrics in performance_metrics.items():
        f.write(f"{name:<20} | {metrics['R2']:<10.4f} | {metrics['RMSE']:<12.4f} | {metrics['MAE']:<10.4f}\n")
    f.write("-" * 65 + "\n\n")

    f.write(f"👉 STRATEJİK SEÇİM NOTU:\n")
    f.write(f"   Yapılan zamansal kronolojik testler sonucunda varyansı en başarılı şekilde çözen\n")
    f.write(f"   ve en yüksek açıklayıcılık gücüne sahip olan '{best_model_name}' şampiyon seçilmiştir.\n")
    f.write(f"   Tedarik Kontrol Kulesi (Procurement Control Tower) üzerindeki tüm dinamik envanter\n")
    f.write(f"   simülasyonları ve güvenlik stoğu tahminleri bu şampiyon modelin çıktıları ile beslenmektedir.\n")
    f.write("======================================================================\n")

print(f"✔ BAŞARILI: Karşılaştırmalı performans liderlik tablosu '{report_path}' dosyasına eklendi.")