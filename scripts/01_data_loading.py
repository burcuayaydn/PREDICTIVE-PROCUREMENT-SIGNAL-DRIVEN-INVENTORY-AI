# ==============================================================================
# PROJE: PREDICTIVE PROCUREMENT & SIGNAL-DRIVEN INVENTORY AI
# MODÜL: 01_DATA_LOADING.PY (RETAILROCKET CONTROL TOWER)
# ==============================================================================

import pandas as pd
import os

# Konsol çıktı düzenlemeleri
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Dinamik Klasör Yapısı (supply_chain/)
script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.dirname(script_dir)

raw_path = os.path.join(base_path, "data", "raw")
processed_path = os.path.join(base_path, "data", "processed")
reports_path = os.path.join(base_path, "outputs", "reports")

os.makedirs(processed_path, exist_ok=True)
os.makedirs(reports_path, exist_ok=True)


def check_df(name, df):
    print(f"\n--- [{name.upper()} TABLOSU İNCELEMESİ] ---")
    print(f"Boyut: {df.shape[0]:,} satır, {df.shape[1]} sütun")
    print("\n🔍 İlk 3 Satır Önizleme:")
    print(df.head(3))
    print("-" * 50)


def load_and_process_retailrocket():
    print("Sistem: 1. Aşama - Ham Verilerin Yüklenmesi Başlıyor...")

    # 1. EVENTS TABLOSU (Kullanıcı Hareketleri - Ana Tablo)
    events_file = os.path.join(raw_path, "events.csv")
    if not os.path.exists(events_file):
        raise FileNotFoundError(f"HATA: 'events.csv' bulunamadı! Lütfen {raw_path} klasörünü kontrol edin.")

    print("-> 'events.csv' yükleniyor (Büyük dosya, lütfen bekleyin)...")
    df_events = pd.read_csv(events_file)

    # Zaman damgasını insani tarihe çevirme
    df_events['date'] = pd.to_datetime(df_events['timestamp'], unit='ms')

    print("\nSistem: 2. Aşama - Veri Optimizasyonu ve Filtreleme...")
    # Sadece analizde ve modellemede kullanacağımız kritik sütunları süzüyoruz
    df_events_cleaned = df_events[['date', 'itemid', 'event', 'transactionid']].copy()
    df_events_cleaned.columns = ['date', 'item_id', 'event_type', 'transaction_id']

    # 3. STRATEJİK İŞ ZEKASI RAPORU (Satın Alma ve Tedarik Odaklı)
    print("Sistem: 3. Aşama - Öncü Sinyal bazlı İş Zekası Raporu Oluşturuluyor...")

    start_date = df_events_cleaned['date'].min()
    end_date = df_events_cleaned['date'].max()
    total_days = (end_date - start_date).days

    event_counts = df_events_cleaned['event_type'].value_counts()
    unique_items = df_events_cleaned['item_id'].nunique()

    # Raporu Yeni Başlığa Uygun Olarak Yazma
    report_file_path = os.path.join(reports_path, "business_insights.txt")
    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write("==================================================================\n")
        f.write("     PREDICTIVE PROCUREMENT & SIGNAL-DRIVEN INVENTORY AI          \n")
        f.write("             SATIN ALMA ERKEN UYARI ÖN RAPORU                     \n")
        f.write("==================================================================\n")
        f.write(f"Rapor Üretim Tarihi: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("--- 1. MAKRO OPERASYONEL SİNYAL ÖZETİ ---\n")
        f.write(
            f"• Analiz Edilen Zaman Aralığı       : {start_date.strftime('%d %B %Y')} - {end_date.strftime('%d %B %Y')}\n")
        f.write(f"• Toplam Kapsanan Gün Sayısı        : {total_days} Gün\n")
        f.write(f"• Takip Altındaki Benzersiz SKU     : {unique_items:,} Farklı Ürün\n\n")

        f.write("--- 2. DAVRANIŞSAL HUNİ (FUNNEL) GİRDİLERİ ---\n")
        for event, count in event_counts.items():
            f.write(f"• {event.upper()} Sinyal Sayısı".ljust(35) + f": {count:,} Kayıt\n")

        f.write("\n--- 3. SATIN ALMA DEPARTMANI İÇİN EN KRİTİK TOP 5 HEDEF ÜRÜN ---\n")
        top_transactions = df_events_cleaned[df_events_cleaned['event_type'] == 'transaction'][
            'item_id'].value_counts().head(5)
        for i, (item_id, count) in enumerate(top_transactions.items(), 1):
            f.write(f"{i}. Ürün ID: {item_id} -> Kısa Sürede {count} Kez Satın Alındı\n")

    print(f"✔ BAŞARILI: Stratejik rapor yeni başlığıyla '{report_file_path}' adresinde güncellendi.")

    # 4. OPTİMİZE EDİLMİŞ VERİYI PARQUET OLARAK KAYDETME
    output_parquet = os.path.join(processed_path, "master_data.parquet")
    df_events_cleaned.to_parquet(output_parquet, index=False)
    print(f"✔ BAŞARILI: Yeni pipeline akışı için '{output_parquet}' kaydedildi.\n")

    # Bilgilendirme Çıktıları
    check_df("Cleaned Signals Data", df_events_cleaned)


if __name__ == "__main__":
    load_and_process_retailrocket()